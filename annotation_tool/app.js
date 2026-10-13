/*
 * app.js — UI of the thread-density annotator v2 (canvas viewer, sidebar, storage).
 * All annotation logic lives in core.js (AnnCore) so it can be tested in node; core.js
 * also runs as a Web Worker for the yarn-orientation estimate (auto-align).
 */
(function () {
  'use strict';
  const C = window.AnnCore;
  const $ = (id) => document.getElementById(id);
  const PREFS_KEY = 'wvaq_fabric_prefs_v1';      // unchanged in v2: same prefs and store
  const STORE_PREFIX = 'wvaq_fabric_v1::';
  const UNDO_MAX = 100;
  const LINE_TOL = 12;   // px (screen) — how close to a guide line a click must be
  const HANDLE_R = 8;    // px (screen) — guide-line handle radius
  const MOVE_OFF = 16;   // px (screen) — move handles sit this far beyond the segment ends
  const ROT_OFF = 40;    // px (screen) — rotation handle of the selected line, beyond its far end
  const FIT_PAD = 48;    // px — room around the window for the handles
  const IDLE_MS = 60000; // input gaps longer than this are idle and never counted as active time
  const EST_N = 512;     // orientation estimate resolution (Nyquist 256 cycles/inch)

  if (!C || typeof C.lineInfo !== 'function' || typeof C.viewToScreen !== 'function' || typeof C.finishedScenes !== 'function') {   // old core.js cached next to a new app.js
    document.body.innerHTML = '<p style="padding:20px;font:15px sans-serif">The annotator was updated while this page ' +
      'was loading. Please reload the page (Cmd/Ctrl-Shift-R). Your work is safe.</p>';
    return;
  }

  // ------------------------------------------------------------------ state
  const st = {
    manifest: null, images: [], byId: {}, order: {}, iaa: new Set(),
    store: C.emptyStore(''), prefs: null,
    cur: null, rec: null,
    imgEl: null, imgKey: null, imgCache: new Map(),
    view: { scale: 1, ox: 0, oy: 0, fit: 1, user: false, rot: 0 },   // user = zoomed/panned since last fit; rot = image turn the offsets were set for
    undo: {}, drag: null, hover: null, spaceDown: false, preview: null,
    serverMode: false, lastLocalSave: null, lastServerSave: null, localOk: true, serverOk: null,
    lastInput: Date.now(), timeDirty: false, pending: 0, pendingId: null,
    est: new Map(), lastUndoTag: null, lastUndoT: 0,
    scenes: false, unlocked: new Set(), ephemeral: false,   // v2.1: scene-aware (manifest scene fields)
    attention: false,   // v2.2: manifest attention fields ("check carefully" badge, "High attention first")
  };

  // ------------------------------------------------------------------ storage (always try/catch)
  function lsGet(k) {
    try { const v = window.localStorage.getItem(k); return v ? JSON.parse(v) : null; } catch (e) { return null; }
  }
  function lsSet(k, v) {
    try { window.localStorage.setItem(k, JSON.stringify(v)); return true; } catch (e) { return false; }
  }
  const DEFAULT_PREFS = {
    annotator: '', blindPolicy: 'iaa', blindManual: false, enhanced: false,
    brightness: 100, contrast: 100, dotSize: 5, showGaps: true, filter: 'scenes', lastId: null, selLine: 'h',
    prefsV: 2,
  };
  function loadPrefs() {
    const saved = lsGet(PREFS_KEY) || {};
    st.prefs = Object.assign({}, DEFAULT_PREFS, saved);
    if (st.prefs.selLine !== 'v') st.prefs.selLine = 'h';
    // v2.1: "Scenes to annotate" becomes the default filter once, also for existing users
    if ((Number(saved.prefsV) || 1) < 2) { st.prefs.filter = 'scenes'; st.prefs.prefsV = 2; }
  }
  function savePrefs() { lsSet(PREFS_KEY, st.prefs); }
  const storeKey = (name) => STORE_PREFIX + C.safeName(name);

  function loadLocalStore(name) {
    const raw = lsGet(storeKey(name));
    if (!raw) return C.emptyStore(name);
    try { const p = C.parseImport(raw); return { schema: C.SCHEMA, annotator: name, records: p.records, updated: raw.updated }; }
    catch (e) { return C.emptyStore(name); }
  }

  let saveTimer = null, serverTimer = null;
  function scheduleSave() {
    clearTimeout(saveTimer);
    saveTimer = setTimeout(saveNow, 250);
  }
  /**
   * Merge records saved by another tab (same annotator) into memory: per record the newer
   * timestamp wins, ties keep ours (so st.rec stays the object in the store).
   */
  function mergeStored(raw) {
    if (!raw || !raw.records) return 0;
    let p;
    try { p = C.parseImport(raw); } catch (e) { return 0; }
    const m = C.mergeRecords(st.store.records, p.records);
    if (!m.added && !m.updated) return 0;
    st.store.records = m.merged;
    if (st.cur) {
      const r = st.store.records[String(st.cur.id)];
      if (r && r !== st.rec) { st.rec = r; renderSide(); requestDraw(); }
    }
    return m.added + m.updated;
  }
  function saveNow() {
    clearTimeout(saveTimer);
    mergeStored(lsGet(storeKey(st.store.annotator)));   // never clobber newer work of another tab
    st.store.updated = new Date().toISOString();
    st.localOk = lsSet(storeKey(st.store.annotator), st.store);
    st.lastLocalSave = new Date();
    st.timeDirty = false;
    if (st.serverMode) { clearTimeout(serverTimer); serverTimer = setTimeout(saveServer, 1200); }
    updateSaveStatus();
  }
  async function saveServer() {
    try {
      const r = await fetch('api/save', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(C.exportJSON(st.store, st.manifest)),
      });
      st.serverOk = r.ok;
      if (r.ok) st.lastServerSave = new Date();
    } catch (e) { st.serverOk = false; }
    updateSaveStatus();
  }
  async function detectServer() {
    try {
      const r = await fetch('api/ping', { cache: 'no-store' });
      if (!r.ok) return false;
      const j = await r.json();
      return !!(j && j.ok);
    } catch (e) { return false; }
  }
  async function loadServerStore(name) {
    try {
      const r = await fetch('api/load?annotator=' + encodeURIComponent(C.safeName(name)), { cache: 'no-store' });
      if (!r.ok) return null;
      return C.parseImport(await r.json());
    } catch (e) { return null; }
  }
  function updateSaveStatus() {
    const el = $('savestatus');
    const t = st.lastLocalSave ? st.lastLocalSave.toLocaleTimeString() : '—';
    let s = st.localOk ? 'saved ' + t + ' · browser ✓' : '⚠ browser storage unavailable — Download often!';
    if (st.serverMode) s += st.serverOk === false ? ' · disk ✗' : ' · disk ✓';
    el.textContent = s;
    el.style.color = st.localOk && st.serverOk !== false ? '' : 'var(--err)';
  }

  // ------------------------------------------------------------------ helpers
  function toast(msg, ms) {
    const el = $('toast');
    el.textContent = msg; el.hidden = false;
    clearTimeout(toast._t);
    toast._t = setTimeout(() => { el.hidden = true; }, ms || 2200);
  }
  const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
  function download(name, text, type) {
    const blob = new Blob([text], { type: type });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob); a.download = name;
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 5000);
  }
  function stamp() {
    const d = new Date(), p = (n) => String(n).padStart(2, '0');
    return d.getFullYear() + p(d.getMonth() + 1) + p(d.getDate()) + '-' + p(d.getHours()) + p(d.getMinutes());
  }
  function fmtDeg(a) {
    if (a == null || !isFinite(a)) return '—';
    const s = Math.abs(a) < 0.05 ? '' : (a > 0 ? '+' : '−');
    return s + Math.abs(a).toFixed(1) + '°';
  }
  function fmtTime(s) {
    s = Math.max(0, Math.round(s));
    const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), ss = String(s % 60).padStart(2, '0');
    return h ? h + ':' + String(m).padStart(2, '0') + ':' + ss : m + ':' + ss;
  }
  const LINE_NAME = { h: 'H', v: 'V' };

  // ------------------------------------------------------------------ active time (idle gaps > 60 s excluded)
  /**
   * Seconds are first accumulated as "pending" while the last input is <= 60 s old, and
   * only credited to the image when the NEXT input arrives within 60 s. If the next input
   * comes later, the whole gap was idle and its pending seconds are dropped, so an idle
   * period longer than 60 s contributes nothing (not even its first minute).
   */
  function commitPending() {
    if (st.pending > 0 && st.pendingId != null) {
      const r = st.store.records[String(st.pendingId)];
      if (r) { r.time_spent_s = (r.time_spent_s || 0) + st.pending; st.timeDirty = true; }
    }
    st.pending = 0; st.pendingId = st.rec ? st.rec.id : null;
  }
  function noteInput() {
    const now = Date.now();
    if (now - st.lastInput <= IDLE_MS) commitPending();
    else { st.pending = 0; st.pendingId = st.rec ? st.rec.id : null; }
    st.lastInput = now;
  }
  /** Leaving the page / tab: credit the seconds since the last input if that gap is not idle. */
  function flushTime() {
    if (Date.now() - st.lastInput <= IDLE_MS) commitPending();
  }
  function tickTime() {
    // read-only copies: no time; v2.2.2: none while the Examples panel is open (not time spent on this image)
    if (st.rec && !readOnly() && !document.hidden && !examplesOpen() && Date.now() - st.lastInput <= IDLE_MS) {
      if (st.pendingId !== st.rec.id) commitPending();
      st.pending++;
    }
    updateTimer();
  }
  function updateTimer() {
    const el = $('timer'); if (!el) return;
    const r = st.rec;
    if (!r) { el.hidden = true; return; }
    const idle = Date.now() - st.lastInput > IDLE_MS || document.hidden || readOnly() || examplesOpen();
    const s = (r.time_spent_s || 0) + (st.pendingId === r.id ? st.pending : 0);
    el.hidden = false;
    el.textContent = '⏱ ' + fmtTime(s) + (idle ? ' · paused' : '');
    el.classList.toggle('idle', idle);
  }

  // ------------------------------------------------------------------ records
  function ensureRec(img) {
    const k = String(img.id);
    const inIaa = st.iaa.has(img.id);
    // an unlocked copy of an IAA scene opens blind too (it shows the same fabric), but is not IAA itself
    const blindIaa = inIaa || (st.scenes && C.isCopy(img) && st.iaa.has(C.sceneRepId(img)));
    let rec = st.store.records[k];
    if (!rec) {
      const blind = C.blindForPolicy(st.prefs.blindPolicy, img.id, st.prefs.blindManual, blindIaa);
      rec = C.newRecord(img, { annotator: st.store.annotator, blind: blind, iaa: inIaa });
      st.store.records[k] = rec;
      scheduleSave();
    } else {
      if (rec.iaa == null) rec.iaa = inIaa;
      // IAA subset re-drawn (v2.1): an opened-but-untouched record follows the new membership
      if (C.syncIaa(rec, img, inIaa, { policy: st.prefs.blindPolicy, manualDefault: st.prefs.blindManual })) scheduleSave();
      // prepare.py --refresh-proposals: untouched records pick up the new proposals;
      // records with any user edit are never changed (a notice offers "Reset to proposals")
      if (C.refreshProposals(rec, img) === 'refreshed') scheduleSave();
    }
    return rec;
  }
  /** v2.1: a non-representative file of a scene that has not been unlocked with "Edit anyway". */
  function readOnly(img) {
    img = img || st.cur;
    return !!(st.scenes && img && C.isCopy(img) && !st.unlocked.has(img.id));
  }
  /** Call at the start of every editing action: true (and a hint) when the image is read-only. */
  function locked() {
    if (!readOnly()) return false;
    const r = C.sceneRepId(st.cur);
    toast('Read-only: same scene as #' + r + ' (labels copied). G opens #' + r + '; "Edit anyway" unlocks this copy.', 3200);
    return true;
  }
  /** Record shown for a read-only copy: its stored record if any, else a blind scratch record never saved. */
  function viewRec(img) {
    const r = st.store.records[String(img.id)];
    st.ephemeral = !r;
    return r || C.newRecord(img, { annotator: st.store.annotator, blind: true, iaa: false });
  }
  function unlockCopy() {
    const img = st.cur;
    if (!img || !readOnly(img)) return;
    const r = C.sceneRepId(img);
    const ok = window.confirm('Image ' + img.id + ' is another file of scene #' + r + ' (' + (C.KIND_LABEL[img.kind] || img.kind) +
      '). Its labels are copied from #' + r + ', so it does not need its own annotation.\n\nEdit this copy anyway? ' +
      'Its own record is kept separately (a repeat count); scoring uses #' + r + '\'s labels whenever #' + r + ' is annotated.');
    if (!ok) return;
    st.unlocked.add(img.id);
    st.rec = ensureRec(img); st.ephemeral = false;
    st.pending = 0; st.pendingId = st.rec.id;
    renderSide(); requestDraw();
    toast('Image ' + img.id + ' unlocked for this session', 2200);
  }
  function goToRep() {
    if (!st.cur) return;
    const r = C.sceneRepId(st.cur);
    if (r === st.cur.id) { toast('This photo is its scene\'s representative'); return; }
    show(r);
  }
  function pushUndo() {
    if (!st.rec) return;
    st.lastUndoTag = null;
    const k = String(st.rec.id);
    const u = st.undo[k] || (st.undo[k] = []);
    u.push(JSON.stringify(st.rec));
    if (u.length > UNDO_MAX) u.shift();
  }
  /** One undo step for a burst of key rotations of the same line. Returns true if it pushed one. */
  function pushUndoCoalesced(tag) {
    const now = Date.now(), key = String(st.rec.id) + ':' + tag;
    if (st.lastUndoTag === key && now - st.lastUndoT < 1500) { st.lastUndoT = now; return false; }
    pushUndo();
    st.lastUndoTag = key; st.lastUndoT = now;
    return true;
  }
  function popUndoIfUnchanged(snapshotLen) {
    const u = st.undo[String(st.rec.id)];
    if (u && u.length > snapshotLen) u.pop();
  }
  function undo() {
    if (!st.rec || locked()) return;
    const u = st.undo[String(st.rec.id)];
    if (!u || !u.length) { toast('Nothing to undo'); return; }
    const rec = JSON.parse(u.pop());
    // keep the audit trail (proposals seen, blind / angle logs, active time) of the current state
    if (typeof C.keepAudit === 'function') C.keepAudit(rec, st.rec);
    C.touch(rec);
    st.store.records[String(rec.id)] = rec;
    st.rec = rec;
    st.lastUndoTag = null;
    commit(false);
  }
  /** Call after any change to st.rec. */
  function commit(withToast) {
    scheduleSave();
    renderSide();
    requestDraw();
    if (withToast) toast(withToast);
  }

  // ------------------------------------------------------------------ navigation / filtering
  function blindOpts() { return { policy: st.prefs.blindPolicy, manualDefault: st.prefs.blindManual }; }
  function filteredIds() {
    return C.filterIds(st.images, st.store.records, st.prefs.filter, st.iaa, st.scenes, blindOpts());
  }
  /** v2.2: "High attention first" is ordered by level, not by id: step through it by list position. */
  function orderedFilter() { return st.attention && st.prefs.filter === 'attention'; }
  function step(dir) {
    const ids = filteredIds();
    if (!ids.length) { toast('No images match this filter'); return; }
    const curPos = st.cur ? st.order[st.cur.id] : -1;
    let target = null;
    if (orderedFilter()) {
      const k = st.cur ? ids.indexOf(st.cur.id) : -1;
      if (k >= 0) target = ids[k + dir] != null ? ids[k + dir] : null;
      else target = dir > 0 ? ids[0] : ids[ids.length - 1];
    } else if (dir > 0) target = ids.find((id) => st.order[id] > curPos);
    else for (let i = ids.length - 1; i >= 0; i--) if (st.order[ids[i]] < curPos) { target = ids[i]; break; }
    if (target == null) {
      if (st.prefs.filter === 'all' || (st.prefs.filter === 'scenes' && !st.scenes)) { toast(dir > 0 ? 'Last image' : 'First image'); return; }
      target = dir > 0 ? ids[0] : ids[ids.length - 1];
      if (st.cur && target === st.cur.id) { toast('No other image in this filter'); return; }
      toast('Wrapped around');
    }
    show(target);
  }
  function nextTodo() {
    const curPos = st.cur ? st.order[st.cur.id] : -1;
    const ids = new Set(C.filterIds(st.images, st.store.records, 'todo', st.iaa, st.scenes));   // copies are never "to do"
    const todo = st.images.filter((im) => ids.has(im.id));
    if (!todo.length) { toast(st.scenes ? 'All scenes are done 🎉' : 'All images are done 🎉'); return; }
    if (orderedFilter()) {   // v2.2: the next unfinished photo in attention order (wraps around)
      const L = filteredIds(), pos = st.cur ? L.indexOf(st.cur.id) : -1;
      const t = L.slice(pos + 1).concat(L.slice(0, pos + 1)).find((id) => ids.has(id));
      if (t != null) { show(t); return; }
    }
    const t = todo.find((im) => st.order[im.id] > curPos) || todo[0];
    show(t.id);
  }

  // ------------------------------------------------------------------ image loading
  function imgSrc(img) { return st.prefs.enhanced ? img.crop_enh : img.crop; }
  function loadImage(src) {
    if (st.imgCache.has(src)) return st.imgCache.get(src);
    const p = new Promise((resolve, reject) => {
      const el = new Image();
      el.onload = () => resolve(el);
      el.onerror = () => reject(new Error('failed to load ' + src));
      el.src = src;
    });
    st.imgCache.set(src, p);
    if (st.imgCache.size > 12) st.imgCache.delete(st.imgCache.keys().next().value);
    p.catch(() => st.imgCache.delete(src));
    return p;
  }
  function preloadNeighbours() {
    if (!st.cur) return;
    const i = st.order[st.cur.id];
    [i + 1, i - 1].forEach((j) => { if (st.images[j]) loadImage(imgSrc(st.images[j])); });
  }

  async function show(id) {
    const img = st.byId[id];
    if (!img) { toast('No image with id ' + id); return; }
    flushTime();                      // credit the previous image's active seconds first
    if (st.rec && st.timeDirty) saveNow();
    st.drag = null; st.hover = null; st.preview = null;
    st.cur = img;
    st.ephemeral = false;
    st.rec = readOnly(img) ? viewRec(img) : ensureRec(img);
    st.pending = 0; st.pendingId = st.rec.id;
    st.prefs.lastId = id; savePrefs();
    $('jump').value = id;
    $('thumb').src = img.thumb;
    fitView();
    renderSide();
    updateTimer();
    await refreshImage(false);
    preloadNeighbours();
    setTimeout(() => { if (st.cur === img) getEstimate(img).then(() => { if (st.cur === img) renderLines(); }, () => renderLines()); }, 60);
  }
  async function refreshImage(fit) {
    const img = st.cur;
    const src = imgSrc(img);
    $('loading').hidden = false;
    try {
      const el = await loadImage(src);
      if (st.cur !== img) return; // user moved on
      st.imgEl = el; st.imgKey = src;
      if (fit) fitView();
    } catch (e) {
      toast(e.message, 4000);
    } finally {
      if (st.cur === img) $('loading').hidden = true;
    }
    requestDraw();
  }

  // ------------------------------------------------------------------ orientation estimate (auto-align)
  let worker = null, workerSeq = 0;
  const workerJobs = new Map();
  function startWorker() {
    if (worker !== null) return worker;
    try {
      worker = new Worker('core.js');
      worker.onmessage = (e) => {
        const d = e.data || {}, job = workerJobs.get(d.id);
        if (!job) return;
        workerJobs.delete(d.id);
        if (d.error) job.reject(new Error(d.error)); else job.resolve(d.result);
      };
      worker.onerror = (e) => {   // e.g. file:// pages: compute on the main thread instead
        if (e && e.preventDefault) e.preventDefault();
        worker = false;
        workerJobs.forEach((job) => { try { job.resolve(C.estimateOrientation(job.gray, job.N, job.opts)); } catch (err) { job.reject(err); } });
        workerJobs.clear();
      };
    } catch (e) { worker = false; }
    return worker;
  }
  function runEstimate(gray, N, opts) {
    const w = startWorker();
    if (!w) return new Promise((res) => setTimeout(res, 0)).then(() => C.estimateOrientation(gray, N, opts));
    return new Promise((resolve, reject) => {
      const id = ++workerSeq;
      workerJobs.set(id, { resolve: resolve, reject: reject, gray: gray, N: N, opts: opts });
      w.postMessage({ id: id, gray: gray, N: N, opts: opts });
    });
  }
  function grayOf(el, N) {
    const cv = document.createElement('canvas');
    cv.width = N; cv.height = N;
    const cx = cv.getContext('2d', { willReadFrequently: true });
    cx.imageSmoothingEnabled = true; cx.imageSmoothingQuality = 'high';
    cx.drawImage(el, 0, 0, N, N);
    const d = cx.getImageData(0, 0, N, N).data, g = new Float32Array(N * N);
    for (let i = 0, j = 0; i < g.length; i++, j += 4) g[i] = 0.299 * d[j] + 0.587 * d[j + 1] + 0.114 * d[j + 2];
    return g;
  }
  /** Promise of the yarn-orientation estimate of an image (computed once, from the plain crop). */
  function getEstimate(img) {
    if (st.est.has(img.id)) return st.est.get(img.id).promise;
    const entry = { status: 'pending', result: null };
    entry.promise = loadImage(img.crop)
      .then((el) => runEstimate(grayOf(el, EST_N), EST_N, { rMax: Math.min(0.9 * img.rect_size[0] / 2, EST_N / 2 - 2) }))
      .then((res) => { entry.status = 'done'; entry.result = res; return res; },
        (err) => { entry.status = 'error'; st.est.delete(img.id); throw err; });
    st.est.set(img.id, entry);
    return entry.promise;
  }
  function estNow(img) {
    const e = img && st.est.get(img.id);
    return e && e.status === 'done' ? e.result : null;
  }
  async function autoAlign(lines) {
    if (!st.rec || !st.cur || locked()) return;
    const img = st.cur;
    let est;
    try { est = await getEstimate(img); } catch (e) { toast('Could not estimate the yarn direction: ' + e.message, 3500); return; }
    if (st.cur !== img || !st.rec) return;
    pushUndo();
    const msgs = [];
    let changed = 0;
    lines.forEach((l) => {
      const e = est[l];
      if (!e) { msgs.push(LINE_NAME[l] + ': no clear yarn direction'); return; }
      if (C.setAngle(st.rec, l, e.deg, 'auto', { est: e.deg, clear: !!e.clear })) changed++;
      msgs.push(LINE_NAME[l] + ' ' + fmtDeg(e.deg) + (e.clear ? '' : ' (weak — check)'));
    });
    if (!changed) popUndoIfUnchanged(0);
    commit('Auto-align: ' + msgs.join(' · ') + (changed ? '' : ' — already aligned'));
  }

  // ------------------------------------------------------------------ view transform
  const imgCanvas = $('imgcanvas'), ovlCanvas = $('ovlcanvas'), stage = $('stage');
  let cssW = 0, cssH = 0, dpr = 1;
  function resizeCanvases() {
    const oldW = cssW, oldH = cssH;
    dpr = window.devicePixelRatio || 1;
    cssW = stage.clientWidth; cssH = stage.clientHeight;
    [imgCanvas, ovlCanvas].forEach((c) => { c.width = Math.round(cssW * dpr); c.height = Math.round(cssH * dpr); });
    // not zoomed/panned by the user, or first real layout (stage was 0 x 0): refit
    if (st.rec && (!st.view.user || !oldW || !oldH)) fitView();
    else { st.view.ox += (cssW - oldW) / 2; st.view.oy += (cssH - oldH) / 2; }    // zoomed: keep centre
    requestDraw();
  }
  /** Fit the whole crop (turned by the image rotation: its bounding box) into the stage, centred. */
  function fitView() {
    if (!st.rec) return;
    if (cssW < 40 || cssH < 40) { cssW = stage.clientWidth; cssH = stage.clientHeight; }
    if (cssW < 40 || cssH < 40) return;  // stage not laid out yet; the ResizeObserver refits
    st.view.user = false;
    st.view.rot = C.viewAngle(st.rec);
    const S = st.rec.rect_size, pad = FIT_PAD, E = S * C.viewExtent(st.view.rot);
    const sc = Math.min((cssW - 2 * pad) / E, (cssH - 2 * pad) / E);
    st.view.scale = st.view.fit = Math.max(0.05, sc);
    st.view.ox = cssW / 2 - S / 2 * st.view.scale;   // the image turns about the crop centre: keep it centred
    st.view.oy = cssH / 2 - S / 2 * st.view.scale;
    requestDraw();
  }
  /**
   * v2.2.1: the record's image turn changed (keys, slider, align, undo, another tab, import): not zoomed
   * -> refit (the turned crop stays fully visible); zoomed -> the image point at the stage centre stays put.
   */
  function syncViewRot() {
    const r = st.rec; if (!r) return;
    const a = C.viewAngle(r), v = st.view;
    if (a === v.rot) return;
    const old = v.rot;
    v.rot = a;
    if (!v.user) { fitView(); return; }
    const S = r.rect_size, cx = cssW / 2, cy = cssH / 2;
    const w = C.screenToWorld({ scale: v.scale, ox: v.ox, oy: v.oy, rot: old }, S, cx, cy);
    const p = C.viewToScreen({ scale: v.scale, ox: v.ox, oy: v.oy, rot: a }, S, w[0], w[1]);
    v.ox += cx - p[0]; v.oy += cy - p[1];
  }
  /** Current view transform {scale, ox, oy, rot} (rot = the record's image turn, deg). */
  function viewOf() {
    syncViewRot();
    return { scale: st.view.scale, ox: st.view.ox, oy: st.view.oy, rot: st.view.rot };
  }
  /** Zoom about screen point (px, py). Works in turned-image coordinates q = (screen - o) / scale, so any image turn is kept. */
  function zoomAt(px, py, f) {
    syncViewRot();
    const v = st.view;
    const ns = clamp(v.scale * f, v.fit * 0.5, 60);
    const qx = (px - v.ox) / v.scale, qy = (py - v.oy) / v.scale;
    v.scale = ns; v.ox = px - qx * ns; v.oy = py - qy * ns; v.user = true;
    requestDraw();
  }
  /** Image point [x, y] -> screen [x, y] (pan, zoom and image turn). */
  const toScreen = (w) => C.viewToScreen(viewOf(), st.rec.rect_size, w[0], w[1]);
  /** Screen point {x, y} -> image [x, y]: the exact inverse, used by every click, drag and hit test. */
  const toWorld = (p) => C.screenToWorld(viewOf(), st.rec.rect_size, p.x, p.y);

  // ------------------------------------------------------------------ line geometry on screen
  /** Screen geometry of a line's segment: start A (lo end), end B, unit direction u, length L. */
  function lineScreen(rec, line) {
    const seg = C.segment(rec, line), u = C.viewDir(viewOf(), C.lineDir(rec, line));   // u on screen (turned with the image)
    const A = toScreen(seg[0]), B = toScreen(seg[1]);
    return { A: A, B: B, u: u, L: Math.hypot(B[0] - A[0], B[1] - A[1]) };
  }
  /** Range of s for which A + s*u lies inside the canvas inset by m px (null if none). */
  function canvasInterval(A, u, m) {
    let lo = -Infinity, hi = Infinity;
    const axes = [[A[0], u[0], m, cssW - m], [A[1], u[1], m, cssH - m]];
    for (let i = 0; i < 2; i++) {
      const [p, d, mn, mx] = axes[i];
      if (Math.abs(d) < 1e-9) { if (p < mn || p > mx) return null; continue; }
      let t1 = (mn - p) / d, t2 = (mx - p) / d;
      if (t1 > t2) { const t = t1; t1 = t2; t2 = t; }
      lo = Math.max(lo, t1); hi = Math.min(hi, t2);
    }
    return lo <= hi ? [lo, hi] : null;
  }
  function handles() {
    const r = st.rec; if (!r || readOnly()) return [];
    const out = [];
    ['h', 'v'].forEach((line) => {
      const g = lineScreen(r, line), iv = canvasInterval(g.A, g.u, 10);
      if (!iv) return;
      const at = (s, kind) => {
        s = clamp(s, iv[0], iv[1]);
        return { line: line, kind: kind, s: s, x: g.A[0] + s * g.u[0], y: g.A[1] + s * g.u[1] };
      };
      const h0 = at(-MOVE_OFF, 'move');
      if (line === st.prefs.selLine) {
        const hr = at(g.L + ROT_OFF, 'rot');
        out.push(h0, at(Math.min(g.L + MOVE_OFF, hr.s - 2 * HANDLE_R - 6), 'move'), hr);
      } else out.push(h0, at(g.L + MOVE_OFF, 'move'));
    });
    return out;
  }

  // ------------------------------------------------------------------ drawing
  let drawQueued = false;
  function requestDraw() {
    if (drawQueued) return;
    drawQueued = true;
    requestAnimationFrame(() => { drawQueued = false; draw(); });
  }
  function draw() {
    const ic = imgCanvas.getContext('2d');
    ic.setTransform(1, 0, 0, 1, 0, 0);
    ic.clearRect(0, 0, imgCanvas.width, imgCanvas.height);
    const v = st.rec ? viewOf() : st.view;
    if (st.imgEl && st.cur && st.imgKey === imgSrc(st.cur)) {
      ic.imageSmoothingEnabled = v.scale < 2.5;
      ic.setTransform(dpr * v.scale, 0, 0, dpr * v.scale, dpr * v.ox, dpr * v.oy);
      const S = st.rec.rect_size;
      if (v.rot) {   // v2.2.1: image turned about the crop centre (same transform as C.viewToScreen)
        ic.translate(S / 2, S / 2); ic.rotate(v.rot * Math.PI / 180); ic.translate(-S / 2, -S / 2);
      }
      ic.drawImage(st.imgEl, 0, 0, S, S);
    }
    imgCanvas.style.filter = 'brightness(' + st.prefs.brightness + '%) contrast(' + st.prefs.contrast + '%)';

    const c = ovlCanvas.getContext('2d');
    c.setTransform(dpr, 0, 0, dpr, 0, 0);
    c.clearRect(0, 0, cssW, cssH);
    const r = st.rec;
    if (!r || !st.cur || st.imgKey !== imgSrc(st.cur)) return; // no overlay until the crop is shown
    const S = r.rect_size, css = getComputedStyle(document.documentElement);
    const col = { acc: css.getPropertyValue('--accent').trim(), h: css.getPropertyValue('--hdot').trim(),
      v: css.getPropertyValue('--vdot').trim(), rot: css.getPropertyValue('--rot').trim() || '#a3e635' };
    const sel = st.prefs.selLine;
    const P = toScreen;

    // window outline (turned with the image, so the tilted 1-inch edges stay visible)
    c.strokeStyle = v.rot ? 'rgba(255,255,255,0.55)' : 'rgba(255,255,255,0.35)'; c.lineWidth = 1; c.setLineDash([4, 4]);
    c.beginPath();
    [[0, 0], [S, 0], [S, S], [0, S]].forEach((w, i) => { const q = P(w); if (i) c.lineTo(q[0], q[1]); else c.moveTo(q[0], q[1]); });
    c.closePath(); c.stroke();
    c.setLineDash([]);

    // suspicious gaps (drawn under the line)
    if (st.prefs.showGaps) {
      ['h', 'v'].forEach((line) => {
        const g = C.lineGapStats(r, line); if (!g) return;
        const seg = (a, b, color) => {
          const p0 = P(C.pointAt(r, line, a)), p1 = P(C.pointAt(r, line, b));
          c.strokeStyle = color; c.lineWidth = 8; c.lineCap = 'butt'; c.beginPath();
          c.moveTo(p0[0], p0[1]); c.lineTo(p1[0], p1[1]); c.stroke();
        };
        g.big.forEach(([a, b]) => seg(a, b, 'rgba(239,68,68,0.55)'));
        g.small.forEach(([a, b]) => seg(a, b, 'rgba(245,158,11,0.75)'));
      });
    }

    // guide lines: segment inside the window, dashed extensions to the handles
    ['h', 'v'].forEach((line) => {
      const g = lineScreen(r, line), isSel = line === sel, far = isSel ? ROT_OFF : MOVE_OFF;
      c.setLineDash([3, 4]); c.lineWidth = 1; c.strokeStyle = col.acc; c.globalAlpha = 0.45;
      c.beginPath();
      c.moveTo(g.A[0] - g.u[0] * MOVE_OFF, g.A[1] - g.u[1] * MOVE_OFF); c.lineTo(g.A[0], g.A[1]);
      c.moveTo(g.B[0], g.B[1]); c.lineTo(g.B[0] + g.u[0] * far, g.B[1] + g.u[1] * far);
      c.stroke(); c.setLineDash([]);
      if (isSel) {   // dark halo keeps the selected line visible on light fabric
        c.globalAlpha = 0.5; c.lineWidth = 3.5; c.strokeStyle = '#000';
        c.beginPath(); c.moveTo(g.A[0], g.A[1]); c.lineTo(g.B[0], g.B[1]); c.stroke();
      }
      c.globalAlpha = isSel ? 1 : 0.85; c.lineWidth = isSel ? 1.6 : 1.25; c.strokeStyle = col.acc;
      c.beginPath(); c.moveTo(g.A[0], g.A[1]); c.lineTo(g.B[0], g.B[1]); c.stroke();
      c.globalAlpha = 1;
    });

    // auto-align preview (while hovering the button): dashed line at the estimated angle
    if (st.preview) {
      const tmp = Object.assign({}, r); tmp[st.preview.line + '_angle_deg'] = st.preview.deg;
      const g = lineScreen(tmp, st.preview.line);
      c.setLineDash([6, 4]); c.lineWidth = 1.5; c.strokeStyle = col.rot;
      c.beginPath(); c.moveTo(g.A[0], g.A[1]); c.lineTo(g.B[0], g.B[1]); c.stroke(); c.setLineDash([]);
    }

    // dots
    const rad = Number(st.prefs.dotSize);
    const hov = st.hover && st.hover.type === 'dot' ? st.hover : null;
    c.font = '10px ui-monospace, Menlo, monospace';
    [['h', col.h], ['v', col.v]].forEach(([line, color]) => {
      const dots = r[line + '_dots'], L = C.lineInfo(r, line);
      for (let i = 0; i < dots.length; i++) {
        const [x, y] = P(C.dotXY(r, line, i));
        if (x < -20 || y < -20 || x > cssW + 20 || y > cssH + 20) continue;
        const isHov = hov && hov.line === line && hov.index === i;
        const outside = L.nOut && (dots[i] < L.lo - 0.5 || dots[i] > L.hi + 0.5);
        c.beginPath(); c.arc(x, y, rad, 0, Math.PI * 2);
        c.fillStyle = isHov ? 'rgba(239,68,68,0.45)' : color + '40';
        c.fill();
        if (outside) c.setLineDash([2, 2]);
        c.lineWidth = isHov ? 2.5 : 1.75; c.strokeStyle = isHov || outside ? '#ef4444' : color; c.stroke();
        c.setLineDash([]);
        if (isHov) { // red X = click deletes
          c.beginPath(); c.moveTo(x - rad * 0.6, y - rad * 0.6); c.lineTo(x + rad * 0.6, y + rad * 0.6);
          c.moveTo(x + rad * 0.6, y - rad * 0.6); c.lineTo(x - rad * 0.6, y + rad * 0.6); c.stroke();
        }
        if ((i + 1) % 10 === 0) {
          c.fillStyle = color; c.textAlign = 'center'; c.textBaseline = 'middle';
          if (line === 'h') c.fillText(String(i + 1), x, y - rad - 8);
          else { c.textAlign = 'right'; c.fillText(String(i + 1), x - rad - 4, y); }
        }
      }
    });

    // ghost dot where a click would add
    if (st.hover && st.hover.type === 'add' && !st.drag) {
      const p = P(C.pointAt(r, st.hover.line, st.hover.pos));
      c.setLineDash([2, 2]); c.lineWidth = 1.5; c.strokeStyle = '#fff';
      c.beginPath(); c.arc(p[0], p[1], rad, 0, Math.PI * 2); c.stroke(); c.setLineDash([]);
    }

    // handles: move (round, arrows) on both ends; rotate (green) beyond the far end of the selected line
    handles().forEach((h) => {
      const active = (st.hover && st.hover.type === 'handle' && st.hover.line === h.line && st.hover.kind === h.kind) ||
        (st.drag && st.drag.line === h.line && ((st.drag.mode === 'line' && h.kind === 'move') || (st.drag.mode === 'rot' && h.kind === 'rot')));
      c.beginPath(); c.arc(h.x, h.y, HANDLE_R, 0, Math.PI * 2);
      c.fillStyle = active ? '#fff' : (h.kind === 'rot' ? col.rot : col.acc); c.fill();
      c.strokeStyle = '#000'; c.lineWidth = 1; c.stroke();
      c.fillStyle = '#000'; c.font = 'bold 10px sans-serif'; c.textAlign = 'center'; c.textBaseline = 'middle';
      c.fillText(h.kind === 'rot' ? '⟳' : (h.line === 'h' ? '↕' : '↔'), h.x, h.y + 0.5);
      if (h.kind === 'rot') {   // angle label next to the rotation handle
        const a = C.lineAngle(r, h.line), txt = LINE_NAME[h.line] + ' ' + fmtDeg(a);
        c.font = '11px ui-monospace, Menlo, monospace';
        const w = c.measureText(txt).width + 8;
        let lx = h.x + 12, ly = h.y - 16;
        if (lx + w > cssW - 4) lx = h.x - 12 - w;
        if (ly < 4) ly = h.y + 6;
        c.fillStyle = 'rgba(0,0,0,0.65)'; c.fillRect(lx, ly, w, 15);
        c.fillStyle = col.rot; c.textAlign = 'left'; c.textBaseline = 'middle'; c.fillText(txt, lx + 4, ly + 8);
      }
    });

    // box selection
    if (st.drag && st.drag.mode === 'box') {
      const d = st.drag;
      c.fillStyle = 'rgba(239,68,68,0.12)'; c.strokeStyle = '#ef4444'; c.lineWidth = 1;
      c.fillRect(Math.min(d.x0, d.x1), Math.min(d.y0, d.y1), Math.abs(d.x1 - d.x0), Math.abs(d.y1 - d.y0));
      c.strokeRect(Math.min(d.x0, d.x1), Math.min(d.y0, d.y1), Math.abs(d.x1 - d.x0), Math.abs(d.y1 - d.y0));
    }
  }

  // ------------------------------------------------------------------ hit testing
  function local(e) {
    const b = stage.getBoundingClientRect();
    return { x: e.clientX - b.left, y: e.clientY - b.top };
  }
  function handleHit(p) {
    let best = null;
    handles().forEach((h) => {
      const d = Math.hypot(p.x - h.x, p.y - h.y);
      if (d <= HANDLE_R + 4 && (!best || d < best.d)) best = { line: h.line, kind: h.kind, d: d };
    });
    return best;
  }
  function dotHit(p) {
    const r = st.rec; if (!r) return null;
    const R = Math.max(Number(st.prefs.dotSize) + 3, 7);
    let best = null;
    ['h', 'v'].forEach((line) => {
      for (let i = 0; i < r[line + '_dots'].length; i++) {
        const q = toScreen(C.dotXY(r, line, i));
        const d = Math.hypot(q[0] - p.x, q[1] - p.y);
        if (d <= R && (!best || d < best.d)) best = { type: 'dot', line: line, index: i, d: d };
      }
    });
    return best;
  }
  /** Nearest line within LINE_TOL screen px of p -> {type:'add', line, pos (projected, on the segment), d}. */
  function lineHit(p) {
    const r = st.rec; if (!r) return null;
    const w = toWorld(p), wx = w[0], wy = w[1], tolW = LINE_TOL / st.view.scale;   // toWorld first: it syncs a pending image turn
    let best = null;
    ['h', 'v'].forEach((line) => {
      const pr = C.projectToLine(r, line, wx, wy), rg = C.paramRange(r, line);
      const d = pr.dist * st.view.scale;
      if (d <= LINE_TOL && pr.raw >= rg.lo - tolW && pr.raw <= rg.hi + tolW && (!best || d < best.d)) {
        best = { type: 'add', line: line, pos: pr.pos, d: d };
      }
    });
    return best;
  }
  function hoverAt(p, e) {
    if ((e && e.shiftKey) || readOnly()) return null;
    const h = handleHit(p);
    if (h) return { type: 'handle', line: h.line, kind: h.kind };
    return dotHit(p) || lineHit(p);
  }
  /** Angle (deg, tool convention) of the direction from the line's pivot to world point (wx, wy). */
  function pointerAngle(line, wx, wy) {
    const pv = C.linePivot(st.rec, line), dx = wx - pv[0], dy = wy - pv[1];
    if (Math.hypot(dx, dy) < 1e-6) return null;
    let a = (line === 'h' ? Math.atan2(dy, dx) : Math.atan2(-dx, dy)) * 180 / Math.PI;
    if (a > 90) a -= 180; else if (a <= -90) a += 180;
    return a;
  }

  // ------------------------------------------------------------------ pointer interaction
  function capture(e) {
    try { stage.setPointerCapture(e.pointerId); } catch (err) { /* synthetic or inactive pointer */ }
  }
  function selectLine(line, quiet) {
    if (line !== 'h' && line !== 'v') return;
    if (st.prefs.selLine !== line) { st.prefs.selLine = line; savePrefs(); }
    if (!quiet) { renderLines(); requestDraw(); }
  }
  function onPointerDown(e) {
    if (e.target && e.target.closest && e.target.closest('#stagebadges button')) return;   // badge buttons (v2.1)
    if (!st.rec || !st.cur || st.imgKey !== imgSrc(st.cur)) return; // crop still loading
    stage.focus({ preventScroll: true });
    const p = local(e);
    const v = st.view;
    if (e.button === 1 || (e.button === 0 && st.spaceDown)) {
      st.drag = { mode: 'pan', x0: p.x, y0: p.y, ox: v.ox, oy: v.oy };
      stage.classList.add('panning');
      capture(e); e.preventDefault(); return;
    }
    if (e.button !== 0) return;
    if (e.shiftKey) {
      if (locked()) return;
      st.drag = { mode: 'box', x0: p.x, y0: p.y, x1: p.x, y1: p.y };
      capture(e); return;
    }
    const h = handleHit(p);
    const lh = !h && e.altKey ? lineHit(p) : null;
    if (lh && locked()) return;
    if (h && h.kind === 'rot') {
      pushUndo();
      const w = toWorld(p), pa = pointerAngle(h.line, w[0], w[1]);
      st.drag = { mode: 'rot', line: h.line, changed: false, off: pa == null ? 0 : C.lineAngle(st.rec, h.line) - pa };
      selectLine(h.line);
      capture(e); e.preventDefault(); return;
    }
    if (h || lh) {
      const line = h ? h.line : lh.line;
      pushUndo();
      const pv = C.linePivot(st.rec, line), w = toWorld(p);
      // keep the grabbed point under the pointer (any line angle and image turn): pivot = pivotThrough(pointer) + off
      st.drag = { mode: 'line', line: line, changed: false,
        off: (line === 'h' ? pv[1] : pv[0]) - C.pivotThrough(st.rec, line, w[0], w[1]) };
      selectLine(line);
      capture(e); e.preventDefault(); return;
    }
    st.drag = { mode: 'press', x0: p.x, y0: p.y, ox: v.ox, oy: v.oy };
    capture(e);
  }
  function onPointerMove(e) {
    noteInput();
    const p = local(e), d = st.drag;
    if (!d) {
      const hv = hoverAt(p, e);
      const key = JSON.stringify(hv), old = JSON.stringify(st.hover);
      st.hover = hv;
      stage.classList.toggle('overhandle', !!(hv && hv.type === 'handle' && hv.kind === 'move'));
      stage.classList.toggle('overrot', !!(hv && hv.type === 'handle' && hv.kind === 'rot'));
      if (key !== old) requestDraw();
      return;
    }
    if (d.mode === 'press' && Math.hypot(p.x - d.x0, p.y - d.y0) > 4) {
      d.mode = 'pan'; stage.classList.add('panning'); st.hover = null;
    }
    if (d.mode === 'pan') {
      st.view.ox = d.ox + (p.x - d.x0); st.view.oy = d.oy + (p.y - d.y0); st.view.user = true; requestDraw();
    } else if (d.mode === 'line') {
      const w = toWorld(p);
      C.moveLine(st.rec, d.line, C.pivotThrough(st.rec, d.line, w[0], w[1]) + d.off);
      d.changed = true; requestDraw(); renderCounts();
    } else if (d.mode === 'rot') {
      const w = toWorld(p), pa = pointerAngle(d.line, w[0], w[1]);
      if (pa != null) {
        const deg = Math.round((pa + d.off) * (e.shiftKey ? 1 : 10)) / (e.shiftKey ? 1 : 10);  // Shift: whole degrees
        if (C.setAngle(st.rec, d.line, deg, 'drag')) { d.changed = true; requestDraw(); renderCounts(); renderLines(); }
      }
    } else if (d.mode === 'box') {
      d.x1 = p.x; d.y1 = p.y; requestDraw();
    }
  }
  function onPointerUp(e) {
    const d = st.drag; if (!d) return;
    st.drag = null;
    stage.classList.remove('panning');
    try { stage.releasePointerCapture(e.pointerId); } catch (err) { /* ignore */ }
    const p = local(e);
    if (d.mode === 'press') clickAt(p);
    else if (d.mode === 'line' || d.mode === 'rot') {
      if (d.changed) commit(); else popUndoIfUnchanged(0);
    } else if (d.mode === 'box') {
      const k = String(st.rec.id), before = (st.undo[k] || []).length;
      pushUndo();
      const n = C.boxDeleteScreen(st.rec, viewOf(), d.x0, d.y0, d.x1, d.y1);   // the box as drawn on screen
      if (n) commit('Deleted ' + n + ' dot' + (n > 1 ? 's' : ''));
      else { popUndoIfUnchanged(before); requestDraw(); }
    }
    st.hover = hoverAt(p, e); requestDraw();
  }
  function clickAt(p) {
    if (readOnly()) { if (lineHit(p) || dotHit(p)) locked(); return; }   // say why, once per click
    const dh = dotHit(p);
    if (dh) { pushUndo(); C.removeDot(st.rec, dh.line, dh.index); selectLine(dh.line, true); commit(); return; }
    const lh = lineHit(p);
    if (lh) { pushUndo(); C.addDot(st.rec, lh.line, lh.pos); selectLine(lh.line, true); commit(); }
  }
  function onWheel(e) {
    e.preventDefault();
    const p = local(e);
    let dy = e.deltaY;
    if (e.deltaMode === 1) dy *= 16; else if (e.deltaMode === 2) dy *= cssH;
    zoomAt(p.x, p.y, Math.exp(-dy * (e.ctrlKey ? 0.01 : 0.0015)));
  }

  // ------------------------------------------------------------------ rotation actions
  function rotateSel(delta) {
    if (!st.rec || locked()) return;
    const l = st.prefs.selLine;
    const before = C.lineAngle(st.rec, l);
    const pushed = pushUndoCoalesced('rot-' + l);
    if (!C.rotateBy(st.rec, l, delta, 'key')) {
      // drop only the snapshot pushed for THIS press; a coalesced burst keeps its own
      if (pushed) { popUndoIfUnchanged(0); st.lastUndoTag = null; }
      toast(LINE_NAME[l] + ' line is at the ' + C.MAX_ANGLE + '° limit');
      return;
    }
    commit();
    if (before === 0 && st.rec[l + '_dots'].length) toast('Rotating keeps each dot\'s ' + (l === 'h' ? 'x' : 'y') + ' position on the line — re-check them', 2600);
  }
  function zeroSel() {
    if (!st.rec || locked()) return;
    const l = st.prefs.selLine;
    pushUndo();
    if (C.setAngle(st.rec, l, 0, 'zero')) commit(LINE_NAME[l] + ' line back to 0°');
    else { popUndoIfUnchanged(0); toast(LINE_NAME[l] + ' line is already at 0°'); }
  }

  // ------------------------------------------------------------------ image (view) rotation (v2.2.1)
  /**
   * Why a view turn of `want` degrees, asked for at image turn `from` (the turn BEFORE it was applied),
   * was cut short: the image limit or a line reaching its 45° limit on the fabric.
   */
  function viewLimitMsg(from, want) {
    const r = st.rec;
    if (Math.abs(from + want) > C.MAX_VIEW + 1e-9) return 'Image is at the ' + C.MAX_VIEW + '° limit';
    return 'Limit: a guide line would pass ' + C.MAX_ANGLE + '° on the fabric (H ' + fmtDeg(C.lineAngle(r, 'h')) +
      ', V ' + fmtDeg(C.lineAngle(r, 'v')) + ')';
  }
  /**
   * Turn the image by `delta` degrees (src: key | slider | zero | align). Rigid with the guide lines:
   * they keep their angle on screen and turn by -delta on the fabric (C.rotateView). One undo step per
   * burst of key presses / slider drag.
   */
  function rotateImage(delta, src, extra) {
    if (!st.rec || locked()) return 0;
    const before = C.viewAngle(st.rec);
    const pushed = src === 'key' || src === 'slider' ? pushUndoCoalesced('view-' + src) : (pushUndo(), true);
    const d = C.rotateView(st.rec, delta, src, extra);
    if (!d) {
      if (pushed) { popUndoIfUnchanged(0); st.lastUndoTag = null; }
      if (Math.abs(delta) > 1e-9) toast(viewLimitMsg(before, delta), 2200);
      renderLines();
      return 0;
    }
    commit();
    if (Math.abs(d - delta) > 0.005) toast(viewLimitMsg(before, delta), 2200);
    else if (before === 0 && (st.rec.h_dots.length || st.rec.v_dots.length) && src !== 'align') {
      toast('The guide lines stay put on screen while the image turns; dots keep their place on the line — re-check them', 3200);
    }
    return d;
  }
  function viewZero() {
    if (!st.rec || locked()) return;
    if (C.viewAngle(st.rec) === 0) { toast('Image is already at 0°'); renderLines(); return; }
    // only claim success when the image really is back at 0 (a line at its 45° limit can stop the turn early;
    // rotateImage has then already said which limit)
    if (rotateImage(-C.viewAngle(st.rec), 'zero') && C.viewAngle(st.rec) === 0) toast('Image back to 0° (the guide lines turned back with it)', 2200);
  }
  /** "Rotate image to align": turn the image until the selected line crosses its yarns at right angles. */
  async function alignImage() {
    if (!st.rec || !st.cur || locked()) return;
    const img = st.cur, l = st.prefs.selLine;
    let est;
    try { est = await getEstimate(img); } catch (e) { toast('Could not estimate the yarn direction: ' + e.message, 3500); return; }
    if (st.cur !== img || !st.rec) return;
    const e = est[l];
    if (!e) { toast(LINE_NAME[l] + ': no clear yarn direction to align the image to', 2600); return; }
    const want = C.viewAlignDelta(st.rec, l, e.deg);
    if (!want) { toast('Image already aligned to the ' + LINE_NAME[l] + ' yarns (' + fmtDeg(e.deg) + ')'); return; }
    const d = rotateImage(want, 'align', { est: e.deg, clear: !!e.clear, ref: l });
    if (d && Math.abs(d - want) <= 0.005) {
      toast('Image turned ' + fmtDeg(d) + ': the ' + LINE_NAME[l] + ' line now crosses its yarns at right angles (estimate ' +
        fmtDeg(e.deg) + (e.clear ? '' : ', weak — check') + ')', 3200);
    }
  }

  // ------------------------------------------------------------------ sidebar
  function chipRow(container, items, onPick, keyHints) {
    container.innerHTML = '';
    items.forEach((it, i) => {
      const b = document.createElement('button');
      b.type = 'button'; b.dataset.key = it.key;
      b.textContent = it.label;
      if (keyHints) { const k = document.createElement('kbd'); k.textContent = String(i + 1); b.appendChild(k); }
      b.addEventListener('click', () => onPick(it.key));
      container.appendChild(b);
    });
  }
  function setChips(container, value) {
    container.querySelectorAll('button').forEach((b) => b.classList.toggle('on', b.dataset.key === String(value)));
  }
  function buildSide() {
    chipRow($('weaves'), C.WEAVES, setWeave, true);
    chipRow($('warpdirs'), C.WARP_DIRS, (k) => setField('warp_dir', k));
    chipRow($('confs'), [{ key: '1', label: '1 low' }, { key: '2', label: '2 medium' }, { key: '3', label: '3 high' }],
      (k) => setField('confidence', Number(k)));
    const fl = $('flags'); fl.innerHTML = '';
    C.FLAGS.forEach((f) => {
      const l = document.createElement('label'); l.className = 'check';
      const cb = document.createElement('input'); cb.type = 'checkbox'; cb.dataset.flag = f.key;
      cb.addEventListener('change', () => {
        if (!st.rec) return;
        if (locked()) { renderSide(); return; }
        pushUndo(); st.rec.flags[f.key] = cb.checked; C.touch(st.rec); commit(); cb.blur();
      });
      l.appendChild(cb); l.appendChild(document.createTextNode(' ' + f.label));
      fl.appendChild(l);
    });
  }
  function setWeave(k) { setField('weave', k); }
  function setField(field, value) {
    if (!st.rec) return;
    if (locked()) { renderSide(); return; }
    pushUndo(); st.rec[field] = value; C.touch(st.rec); commit();
  }
  function cycle(field, values) {
    if (locked()) return;
    const i = values.indexOf(st.rec[field]);
    setField(field, values[(i + 1) % values.length]);
  }
  function gapText(g, n) {
    if (!g) return n ? 'need ≥ 3 dots for spacing check' : '';
    let s = 'spacing ≈ ' + g.est.toFixed(0) + '/in';
    if (g.big.length) s += ' · <span class="bad">' + g.big.length + ' gap' + (g.big.length > 1 ? 's' : '') + '</span>';
    if (g.small.length) s += ' · <span class="meh">' + g.small.length + ' tight</span>';
    return s;
  }
  function fmtTpi(L) {
    if (L.tpi == null) return '—';
    if (L.angle === 0 || Math.abs(L.tpi - Math.round(L.tpi)) < 0.005) return String(Math.round(L.tpi));
    return L.tpi.toFixed(1);
  }
  /** Count card: normalised TPI (big), raw dots + segment length + angle, spacing check. */
  function renderCounts() {
    const r = st.rec; if (!r) return;
    if (readOnly()) { renderCopiedCounts(); return; }
    let out = 0;
    ['h', 'v'].forEach((l) => {
      const L = C.lineInfo(r, l);
      $(l + 'count').textContent = fmtTpi(L);
      let raw = L.n + ' dot' + (L.n === 1 ? '' : 's') + ' / ' + L.segIn.toFixed(3) + ' in';
      if (L.angle !== 0) raw += ' · ' + fmtDeg(L.angle);
      if (L.nOut) raw += ' · <span class="bad">' + L.nOut + ' outside</span>';
      $(l + 'raw').innerHTML = raw;
      $(l + 'sub').innerHTML = gapText(C.lineGapStats(r, l), L.n);
      out += L.nOut;
      $('count' + l).classList.toggle('sel', st.prefs.selLine === l);
    });
    const ow = $('outwarn');
    ow.hidden = !out;
    if (out) $('outwarntext').textContent = out + ' dot' + (out > 1 ? 's lie' : ' lies') + ' outside the window after rotating (not counted in TPI).';
    const ww = C.warpWeftTpi(r);
    const f = (x) => (Math.abs(x - Math.round(x)) < 0.005 ? String(Math.round(x)) : x.toFixed(1));
    $('warpweft').innerHTML = ww.warp != null
      ? 'Warp ends/inch <b>' + f(ww.warp) + '</b> · Weft picks/inch <b>' + f(ww.weft) + '</b>'
      : '<span class="muted">Set warp direction to get warp / weft counts</span>';
  }
  /** v2.1 count card of a read-only copy: the rep's threads/inch as they apply to this file. */
  function renderCopiedCounts() {
    const img = st.cur, repId = C.sceneRepId(img), repRec = st.store.records[String(repId)];
    // an opened-but-untouched rep record only holds machine proposals: nothing to copy yet
    const cl = repRec && !C.isPristine(repRec) ? C.copiedLabel(repRec, img) : null;
    const f = (x) => (x == null ? '—' : Math.abs(x - Math.round(x)) < 0.005 ? String(Math.round(x)) : x.toFixed(1));
    const turn = { 1: '90° counter-clockwise', 2: '180°', 3: '90° clockwise' }[C.sceneRotK(img)];
    ['h', 'v'].forEach((l) => {
      $(l + 'count').textContent = cl ? f(cl[l]) : '—';
      $(l + 'raw').innerHTML = cl ? 'copied from #' + repId + (cl.done ? '' : ' <span class="meh">(not done yet)</span>')
        : '<span class="muted">#' + repId + ' not counted yet</span>';
      $(l + 'sub').innerHTML = '';
      $('count' + l).classList.remove('sel');
    });
    $('outwarn').hidden = true;
    let t = turn ? 'This photo is #' + repId + ' turned ' + turn + (C.sceneRotK(img) % 2 ? ', so H and V are swapped.' : '.') : 'Same orientation as #' + repId + '.';
    if (cl && cl.warp_dir && cl.warp_dir !== 'unknown') t += ' Warp: ' + cl.warp_dir + '.';
    $('warpweft').innerHTML = '<span class="muted">' + t + '</span>';
  }
  /** Guide-line card: selected line, its angle, the image estimate and the auto-align button. */
  function renderLines() {
    const r = st.rec, img = st.cur; if (!r || !img) return;
    const l = st.prefs.selLine;
    $('selh').classList.toggle('on', l === 'h');
    $('selv').classList.toggle('on', l === 'v');
    $('angleval').textContent = LINE_NAME[l] + ' ' + fmtDeg(C.lineAngle(r, l));
    $('angleother').textContent = (l === 'h' ? 'V ' : 'H ') + fmtDeg(C.lineAngle(r, l === 'h' ? 'v' : 'h'));
    const e = st.est.get(img.id), est = estNow(img);
    const desc = (x) => (x ? fmtDeg(x.deg) + (x.clear ? '' : ' <span class="meh">weak</span>') : '<span class="muted">none</span>');
    let t;
    if (est) t = 'Yarn-direction estimate: H ' + desc(est.h) + ' · V ' + desc(est.v);
    else if (e && e.status === 'pending') t = '<span class="muted">Estimating yarn directions…</span>';
    else t = '<span class="muted">Yarn-direction estimate unavailable</span>';
    const p = img.proposal || {};
    if (!r.blind_mode && (isFinite(p.h_angle_deg) || isFinite(p.v_angle_deg))) {
      t += '<br><span class="muted">Method proposal: H ' + fmtDeg(Number(p.h_angle_deg)) + ' · V ' + fmtDeg(Number(p.v_angle_deg)) + '</span>';
    }
    $('estinfo').innerHTML = t;
    const target = est && est[l];
    const btn = $('autoalign');
    btn.disabled = !target || readOnly();
    btn.innerHTML = 'Auto-align ' + LINE_NAME[l] + (target ? ' → ' + fmtDeg(target.deg) : '') + ' <kbd>A</kbd>';
    btn.title = target ? 'Set the ' + LINE_NAME[l] + ' line perpendicular to the yarns it crosses (' + fmtDeg(target.deg) +
      (target.clear ? '' : ', weak estimate — check') + '). Shift-A aligns both lines.' : 'No estimate yet';
    // v2.2.1: image turn
    const va = C.viewAngle(r);
    $('viewval').textContent = fmtDeg(va).replace('—', '0.0°');
    if (document.activeElement !== $('viewrot')) $('viewrot').value = String(va);
    const vb = $('viewalign');
    vb.disabled = !target || readOnly();
    vb.innerHTML = 'Align image to ' + LINE_NAME[l] + ' <kbd>R</kbd>';
    vb.title = target ? 'Turn the image so that the ' + LINE_NAME[l] + ' line crosses its yarns at right angles (estimate ' +
      fmtDeg(target.deg) + (target.clear ? '' : ', weak — check') + '); the guide lines keep their place on screen.' : 'No estimate yet';
  }
  function renderSide() {
    const r = st.rec, img = st.cur;
    if (!r || !img) return;
    const S = r.rect_size;
    const pg = C.progress(st.images, st.store.records, st.iaa, st.scenes);
    $('progtext').textContent = pg.done + ' / ' + pg.total + (pg.unit === 'scenes' ? ' scenes done' : ' done') +
      ' · IAA ' + pg.iaaDone + ' / ' + pg.iaaTotal;
    $('progbar').style.width = (100 * pg.done / Math.max(1, pg.total)) + '%';
    const ids = filteredIds(), pos = ids.indexOf(img.id);
    $('navpos').textContent = pos >= 0 ? (pos + 1) + ' of ' + ids.length : '(not in filter)';

    const ro = readOnly(), copy = st.scenes && C.isCopy(img), repId = C.sceneRepId(img);
    let info = '<b>Image ' + img.id + '</b> · ' + img.file + ' · ' + img.orig_w + '×' + img.orig_h +
      '<br>crop ' + S + '×' + S + ' px = 1 inch (' + S + ' px/inch)';
    if (st.scenes) {
      const others = (img.scene_members || []).filter((m) => m !== img.id);
      info += copy
        ? '<br>Scene #' + repId + ' · ' + (C.KIND_LABEL[img.kind] || img.kind) + (ro ? ' · read-only' : ' · <b>unlocked</b>')
        : '<br>Scene #' + repId + ' · ' + (others.length ? 'labels are copied to ' + others.map((m) => '#' + m).join(', ') : 'single photo');
      if (copy && !st.ephemeral) info += '<br><span class="muted">This copy has its own record (' + st.rec.h_dots.length + ' + ' +
        st.rec.v_dots.length + ' dots' + (st.rec.done ? ', done' : '') + '); it is kept, and #' + repId + '\'s labels take precedence.</span>';
    }
    if (st.iaa.has(img.id)) info += '<span class="tag iaa">IAA subset</span>';
    if (r.done && !st.ephemeral) info += '<span class="tag" style="background:#1f5f46">done</span>';
    // v2.2: "check carefully" (never on IAA-subset scenes, blind records or copies; never a count)
    const attn = st.attention && !st.ephemeral && C.attentionVisible(img, r, st.iaa, blindOpts()) ? C.attentionDisplay(img) : null;
    $('imginfo').innerHTML = info;
    if (attn) {
      const line = document.createElement('div');
      line.className = 'attnline ' + attn.level;
      line.title = 'From the automatic estimators and capture checks only (no counts are shown). Count every yarn as usual.';
      line.textContent = (attn.level === 'high' ? 'Check carefully: ' : 'Take care: ') + attn.line;
      $('imginfo').appendChild(line);
    }
    const ab = $('attnbadge');
    if (ab) {
      ab.hidden = !attn;
      ab.textContent = attn ? attn.badge : '';
      ab.className = 'badge attn' + (attn ? ' ' + attn.level : '');
    }
    $('scenebadge').hidden = !copy;
    if (copy) {
      $('scenetext').textContent = 'same scene as #' + repId + ' — labels copied';
      $('scenego').textContent = 'Go to #' + repId;
      $('sceneunlock').hidden = !ro;
    }
    document.querySelectorAll('#side .edit button, #side .edit input, #side .edit textarea').forEach((el) => { el.disabled = ro; });
    $('side').classList.toggle('readonly', ro);

    renderCounts();
    renderLines();

    $('blind').checked = !!r.blind_mode;
    $('blindpolicy').value = st.prefs.blindPolicy;
    $('blindbadge').hidden = !r.blind_mode || st.ephemeral;
    $('blindbadge').textContent = r.blind_mode && r.proposals_seen
      ? 'BLIND MODE — but proposals were already seen (recorded)' : 'BLIND MODE — proposals hidden';
    $('iaabadge').hidden = !st.iaa.has(img.id);
    $('donebadge').hidden = !r.done || st.ephemeral;
    const p = img.proposal || {};
    if (ro) {
      $('propinfo').textContent = 'Read-only copy of scene #' + repId + ': annotate #' + repId + ' instead (G).' +
        (st.ephemeral ? ' Proposals are not shown on copies.' : '');
    } else if (r.blind_mode) {
      $('propinfo').textContent = r.proposals_seen
        ? 'Proposals hidden, but this image was opened with proposals before (recorded as not blind-clean).'
        : 'Proposals hidden — count every yarn from scratch.';
    } else {
      const kept = (a) => a.reduce((s, x) => s + (x ? 1 : 0), 0);
      let t = 'Proposals: ' + (p.method || 'none') + ' — H ' + (p.h_positions || []).length +
        ', V ' + (p.v_positions || []).length + ' (still unedited: ' + kept(r.h_orig) + ' / ' + kept(r.v_orig) + ')';
      if (r.proposal_sig && r.proposal_sig !== C.proposalSig(img)) {
        t += ' · Newer proposals exist (' + (p.method || '?') + '); your dots are kept — "Reset to proposals" switches.';
      }
      $('propinfo').textContent = t;
    }
    const warn = $('warnbadge');
    if (r.rect_size !== img.rect_size[0]) {
      warn.hidden = false;
      warn.textContent = 'Crop size changed since this annotation (' + r.rect_size + ' → ' + img.rect_size[0] + ' px) — check dots';
    } else warn.hidden = true;

    setChips($('weaves'), r.weave);
    setChips($('warpdirs'), r.warp_dir);
    setChips($('confs'), r.confidence);
    $('flags').querySelectorAll('input').forEach((cb) => { cb.checked = !!(r.flags && r.flags[cb.dataset.flag]); });
    $('nonwoven').checked = !!r.nonwoven;
    if (document.activeElement !== $('notes')) $('notes').value = r.notes || '';
  }

  // ------------------------------------------------------------------ actions
  function doneProblems(r) {   // enforce-done v1
    const p = [];
    if (!r.weave) p.push('choose the weave ("Unsure" is fine)');
    if (!r.confidence) p.push('set the confidence');
    const S = Number(r.rect_size) || 1;
    const why = String(r.notes || '').trim().length > 0;
    const occl = !!(r.flags && r.flags.occluded);
    [['h', 'H line'], ['v', 'V line']].forEach(([ax, name]) => {
      const d = (r[ax + '_dots'] || []).map(Number).filter((x) => isFinite(x));
      if (!d.length) {
        if (!(why && Number(r.confidence) === 1))
          p.push(name + ' has no dots (if its yarns cannot be seen anywhere: write why in Notes and set confidence 1)');
        return;
      }
      const span = (Math.max.apply(null, d) - Math.min.apply(null, d)) / S;
      if (span < 0.75 && !occl && !why)
        p.push(name + ' is only partly counted (dots cover ' + Math.round(span * 100) + '% of it): count to both ends, ' +
               'or tick "Partially occluded" / write why in Notes');
      if (d.length < 10) { p.push(name + ' has only ' + d.length + ' dots: put one dot on every yarn'); return; }
      const s = d.slice().sort((a, b) => a - b), g = s.slice(1).map((x, i) => x - s[i]);
      const med = g.slice().sort((a, b) => a - b)[Math.floor(g.length / 2)];
      const big = g.filter((x) => x > 3 * med).length;
      if (big && !occl && !why)
        p.push(name + ' has ' + big + ' big gap' + (big > 1 ? 's' : '') + ' (red bar): add the missed yarns, ' +
               'or tick "Partially occluded" / write why in Notes');
    });
    return p;
  }
  function saveDone(goNext) {
    if (!st.rec) return;
    if (readOnly()) {   // Enter on a copy: nothing to save; move on (copies are never "to do")
      if (goNext) step(1); else locked();
      return;
    }
    const r = st.rec;
    const missing = [];
    if (!r.warp_dir) missing.push('warp direction');
    const block = doneProblems(r);   // enforce-done v1: a photo is done only when it is complete
    if (block.length) {
      if (r.done) { pushUndo(); r.done = false; C.touch(r); saveNow(); renderSide(); requestDraw(); }
      toast('Not marked done: ' + block.join(' · '), 7000);
      return;
    }
    if (!r.done) { pushUndo(); r.done = true; C.touch(r); }
    saveNow(); renderSide(); requestDraw();
    toast('Saved image ' + r.id + ' ✓' + (missing.length ? ' (not set: ' + missing.join(', ') + ')' : ''),
      missing.length ? 3000 : 1400);
    if (goNext) step(1);
  }
  function toggleBlind() {
    if (!st.rec) return;
    if (locked()) { renderSide(); return; }
    const r = st.rec;
    if (r.blind_mode) {
      const own = r.h_orig.concat(r.v_orig).filter((o) => !o).length;
      const msg = 'Turn blind mode OFF for image ' + r.id + '?\n\nThe machine proposals will be shown' +
        (own ? ' and merged into your ' + own + ' dots (near-duplicates are skipped, so re-check the count)' : '') +
        '. This image will be recorded as "proposals seen", so it no longer counts as a blind annotation.' +
        (st.iaa.has(r.id) ? '\n\nThis is an IAA-subset image: the protocol asks for a blind count.' : '');
      if (!window.confirm(msg)) { renderSide(); return; }
    }
    pushUndo();
    C.setBlind(r, st.cur, !r.blind_mode);
    if (st.prefs.blindPolicy === 'manual') { st.prefs.blindManual = r.blind_mode; savePrefs(); }
    commit(r.blind_mode
      ? (r.proposals_seen ? 'Blind mode ON — proposals removed (recorded: they were already seen)' : 'Blind mode ON — proposals hidden')
      : 'Blind mode OFF — proposals shown');
  }
  function exportAll() {
    if (st.rec) saveNow();
    const name = C.safeName(st.store.annotator);
    const js = JSON.stringify(C.exportJSON(st.store, st.manifest), null, 1);
    download('annotations_' + name + '_' + stamp() + '.json', js, 'application/json');
    setTimeout(() => download('annotations_' + name + '_' + stamp() + '.csv', C.exportCSV(st.store), 'text/csv'), 400);
    toast('Downloading JSON + CSV (' + Object.keys(st.store.records).length + ' records)');
  }
  async function importFile(file) {
    let parsed;
    try {
      const text = await file.text();
      const isJson = /\.json$/i.test(file.name) || /^\s*[{[]/.test(text);
      parsed = isJson ? C.parseImport(JSON.parse(text)) : C.parseImportCSV(text);
    } catch (e) { toast('Import failed: ' + e.message, 4000); return; }
    const fileName = parsed.annotator || '';
    if (fileName && C.safeName(fileName) !== C.safeName(st.store.annotator)) {
      const ok = window.confirm('This file was annotated by "' + fileName + '" but the current annotator is "' +
        (st.store.annotator || '(none)') + '".\n\nOK = switch to "' + fileName + '" and import.\nCancel = abort.');
      if (!ok) return;
      await switchAnnotator(fileName);
    }
    Object.values(parsed.records).forEach((r) => { if (!r.annotator) r.annotator = st.store.annotator; });
    const m = C.mergeRecords(st.store.records, parsed.records);
    st.store.records = m.merged;
    st.undo = {};
    saveNow();
    if (st.cur) { st.rec = ensureRec(st.cur); renderSide(); requestDraw(); }
    toast('Imported: ' + m.added + ' new, ' + m.updated + ' updated, ' + m.kept + ' unchanged (local copy same or newer)', 4000);
  }
  /**
   * v2.2.2: while a name switch loads (the disk autosave is awaited), the Examples panel shows neither the previous
   * name's finished scenes nor figures checked against its records; it is re-rendered for the new name afterwards.
   */
  async function switchAnnotator(name) {
    st.switching = (st.switching || 0) + 1;
    if (examplesOpen()) exRefresh();
    try { await switchAnnotatorNow(name); } finally {
      st.switching--;
      if (examplesOpen()) exRefresh();
    }
  }
  /** Ask before unnamed annotations are filed under `name` (no dot, count or image of them is shown here). */
  function confirmAdopt(edited, name) {
    const n = edited.length;
    const last = edited.reduce((t, r) => Math.max(t, Date.parse(r.timestamp) || 0), 0);
    return window.confirm(n + ' annotation' + (n > 1 ? 's were' : ' was') + ' made in this tool without an annotator name' +
      (last ? ' (last edit ' + new Date(last).toLocaleString() + ')' : '') + '.\n\nAre they YOUR work?\n\n' +
      'OK = yes: file them under "' + name + '".\n' +
      'Cancel = no: leave them unnamed; you start with only your own records. Never take over another ' +
      'person\'s work (their counts are not blind for you).');
  }
  async function switchAnnotatorNow(name) {
    name = String(name || '').trim();
    if (st.rec) saveNow();
    const prev = st.store;
    let next = loadLocalStore(name);
    // first time a name is typed: adopt the work done while the name was empty -- v2.2.2: only when this person
    // confirms it is theirs. The unnamed records may be another person's (e.g. loaded from the unnamed disk autosave,
    // annotations_anonymous.json, in a fresh browser); taking them over silently would show that person's dots, also
    // blind IAA counts, under this name (main view and "My finished scenes").
    const edited = Object.values(prev.records).filter((r) => !C.isPristine(r));
    let adopt = !prev.annotator && !!name && edited.length > 0 && !Object.keys(next.records).length;
    if (adopt && !confirmAdopt(edited, name)) { adopt = false; toast('The unnamed annotations stay unnamed', 3000); }
    if (adopt) {
      next = { schema: C.SCHEMA, annotator: name, records: prev.records, updated: prev.updated };
      Object.values(next.records).forEach((r) => { r.annotator = name; });
      try { window.localStorage.removeItem(storeKey('')); } catch (e) { /* ignore */ }
      toast('Your unnamed annotations now belong to "' + name + '"', 3000);
    }
    next.annotator = name;
    if (st.serverMode) {
      const srv = await loadServerStore(name);
      if (srv) next.records = C.mergeRecords(next.records, srv.records).merged;
    }
    st.store = next; st.undo = {};
    st.prefs.annotator = name; savePrefs();
    $('annotator').value = name;
    saveNow();
    if (st.cur) { st.rec = ensureRec(st.cur); st.pendingId = st.rec.id; renderSide(); requestDraw(); }
  }
  function setHelp(open) {
    $('help').hidden = !open;
    if (open) $('helpclose').focus({ preventScroll: true });
  }

  // ------------------------------------------------------------------ examples panel (v2.2.2, key X)
  // "How it looks": figures listed in guide/examples/examples.json (make_tool_examples.py; non-IAA scenes only,
  // re-checked here by C.visibleExamples, which also hides a figure whose scene this annotator counts blind).
  // "My finished scenes": C.finishedScenes = only the current annotator's own done records; IAA-subset scenes show
  // neither dots nor counts. Read-only: nothing here changes a record. The grid is emptied whenever the panel closes.
  const EX_INDEX = 'guide/examples/examples.json';
  const THUMB_PX = 320;   // downscaled crop cached per file (plain photo, no dots: the same for every annotator)
  const ex = { index: null, err: null, loading: null, tab: 'look', small: new Map(), queue: [], busy: 0, obs: null, gen: 0 };
  function examplesOpen() { const el = $('examples'); return !!(el && !el.hidden); }
  function setExamples(open, tab) {
    const el = $('examples');
    if (!open) {
      if (el.hidden) return;
      el.hidden = true;
      clearGallery();
      stage.focus({ preventScroll: true });
      updateTimer();
      return;
    }
    flushTime();   // credit the image's active seconds before the panel pauses the timer
    setHelp(false);
    el.hidden = false;
    exTab(tab || ex.tab);
    updateTimer();
  }
  function exTab(tab) {
    ex.tab = tab === 'mine' ? 'mine' : 'look';
    const look = ex.tab === 'look';
    $('extab-look').setAttribute('aria-selected', String(look));
    $('extab-mine').setAttribute('aria-selected', String(!look));
    $('expane-look').hidden = !look;
    $('expane-mine').hidden = look;
    (look ? $('extab-look') : $('extab-mine')).focus({ preventScroll: true });
    if (look) { clearGallery(); renderLook(); } else renderMine();
  }
  function loadExampleIndex() {
    if (ex.index) return Promise.resolve(ex.index);
    if (!ex.loading) {
      ex.loading = fetch(EX_INDEX, { cache: 'no-store' })
        .then((r) => { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); })
        .then((j) => { ex.index = j; ex.err = null; return j; }, (e) => { ex.err = e.message || String(e); return null; })
        .then((j) => { ex.loading = null; return j; });
    }
    return ex.loading;
  }
  /** A name switch is loading, or the Annotator field differs from the loaded store: show nothing per-annotator yet. */
  function annotatorPending() {
    return !!st.switching || String($('annotator').value || '').trim() !== String(st.store.annotator || '').trim();
  }
  function exRefresh() { if (ex.tab === 'look') renderLook(); else renderMine(); }
  function renderLook() {
    const box = $('exfigs'), status = $('exstatus');
    if (annotatorPending()) { box.innerHTML = ''; status.textContent = 'Switching annotator…'; return; }
    if (!ex.index) {
      box.innerHTML = '';
      status.textContent = 'Loading the example figures…';
      loadExampleIndex().then((j) => {
        if (!examplesOpen() || ex.tab !== 'look') return;
        if (j || annotatorPending()) renderLook();
        else status.textContent = 'Example figures unavailable (' + ex.err + '). The tips above still apply.';
      });
      return;
    }
    const v = C.visibleExamples(ex.index, st.images, st.store.records, st.iaa, blindOpts());
    box.innerHTML = '';
    v.shown.forEach((f) => {
      const fig = document.createElement('figure');
      const im = document.createElement('img');
      im.loading = 'lazy'; im.decoding = 'async'; im.alt = String(f.title || 'example');
      if (Array.isArray(f.px) && f.px[0] > 0 && f.px[1] > 0) { im.width = f.px[0]; im.height = f.px[1]; }   // reserve the space
      im.src = f.src;
      const cap = document.createElement('figcaption');
      const b = document.createElement('b'); b.textContent = String(f.title || '');
      const c = document.createElement('span'); c.className = 'cap'; c.textContent = ' · ' + String(f.caption || '');
      cap.appendChild(b); cap.appendChild(c);
      fig.appendChild(im); fig.appendChild(cap);
      box.appendChild(fig);
    });
    const n = v.blind + v.refused;
    status.textContent = n ? n + ' figure' + (n > 1 ? 's are' : ' is') + ' not shown' +
      (v.blind ? ' (' + v.blind + ' because ' + (v.blind > 1 ? 'their photos open' : 'its photo opens') +
        ' blind for you: proposals are never shown before a blind count)' : '') + '.' : '';
  }
  function clearGallery() {
    ex.gen++; ex.queue = [];
    if (ex.obs) { ex.obs.disconnect(); ex.obs = null; }
    $('minegrid').innerHTML = ''; $('minehead').innerHTML = '';
  }
  function fmtAgo(t) {
    if (!t) return '';
    const s = (Date.now() - t) / 1000;
    if (s < 60) return 'just now';
    if (s < 3600) return Math.floor(s / 60) + ' min ago';
    if (s < 86400) return Math.floor(s / 3600) + ' h ago';
    return new Date(t).toLocaleDateString();
  }
  function fmtTpiVal(t, a) {
    if (t == null || !isFinite(t)) return '—';
    return a === 0 || Math.abs(t - Math.round(t)) < 0.005 ? String(Math.round(t)) : t.toFixed(1);
  }
  function renderMine() {
    clearGallery();
    const head = $('minehead'), grid = $('minegrid'), name = st.store.annotator;
    if (annotatorPending()) {
      const d = document.createElement('div'); d.className = 'needname'; d.textContent = 'Switching annotator…';
      grid.appendChild(d);
      return;
    }
    const g = C.finishedScenes(st.store.records, name, st.images, st.iaa);
    if (g.needName) {
      const d = document.createElement('div'); d.className = 'needname';
      d.innerHTML = 'Type your name in <b>Annotator</b> (top bar) to see your finished scenes.<br>' +
        '<span class="muted">Only your own records are shown here, never another annotator&rsquo;s.</span><br>';
      const b = document.createElement('button'); b.type = 'button'; b.className = 'primary'; b.id = 'minename';
      b.textContent = 'Enter my name';
      b.onclick = () => { setExamples(false); $('annotator').focus(); };
      d.appendChild(b); grid.appendChild(d);
      return;
    }
    const n = document.createElement('span'); n.className = 'n'; n.textContent = g.nDone + ' done / ' + g.total;
    const who = document.createElement('span'); who.innerHTML = (st.scenes ? 'scenes' : 'images') + ' · annotator <b></b> · most recent first';
    who.querySelector('b').textContent = name;
    const note = document.createElement('span'); note.className = 'muted';
    note.textContent = 'Only records saved under this name. IAA-subset scenes show no dots or counts here. Click one to open it.';
    head.appendChild(n); head.appendChild(who); head.appendChild(note);
    if (!g.items.length) {
      const d = document.createElement('div'); d.className = 'needname';
      d.textContent = 'No finished scenes yet for this name. Press S (done) or Enter (done + next) on a scene.';
      grid.appendChild(d);
      return;
    }
    const cvs = g.items.map((it) => { const t = tileEl(it); grid.appendChild(t); return t.querySelector('canvas'); });
    if ('IntersectionObserver' in window) {
      ex.obs = new IntersectionObserver((ents) => ents.forEach((e) => {
        if (e.isIntersecting && ex.obs) { ex.obs.unobserve(e.target); queueThumb(e.target); }
      }), { root: $('examples').querySelector('.modal-body'), rootMargin: '300px' });
      cvs.forEach((cv) => ex.obs.observe(cv));
    } else cvs.forEach(queueThumb);
  }
  function tileEl(it) {
    const b = document.createElement('button');
    b.type = 'button'; b.className = 'tile' + (it.empty.length ? ' cleared' : '');
    b.dataset.id = String(it.id);
    b.title = 'Open #' + it.id + (it.copy ? ' (a copy of scene #' + it.rep + ')' : '');
    const cv = document.createElement('canvas'); cv._it = it; b.appendChild(cv);
    const add = (cls, html) => { const d = document.createElement('div'); d.className = cls; d.innerHTML = html; b.appendChild(d); return d; };
    if (it.empty.length) {
      const m = document.createElement('span'); m.className = 'clearmark';
      m.textContent = it.empty.map((l) => LINE_NAME[l]).join(' + ') + ' empty';
      m.title = 'No dots on the ' + it.empty.map((l) => LINE_NAME[l]).join(' and ') + ' line (cleared?)';
      b.appendChild(m);
    }
    add('t1', '<span><b>#' + it.id + '</b>' + (it.copy ? ' <span class="tag">copy of #' + it.rep + '</span>' : '') +
      (it.iaa ? ' <span class="tag iaa">IAA</span>' : '') + '</span><span class="muted">' + fmtAgo(it.t) + '</span>');
    if (it.iaa) add('iaanote', 'IAA subset: counted blind &mdash; dots and counts are not shown here');
    else {
      add('tpi', 'H <span class="h">' + fmtTpiVal(it.tpi.h, it.angle.h) + '</span> · V <span class="v">' +
        fmtTpiVal(it.tpi.v, it.angle.v) + '</span> <span class="muted">/in</span>');
    }
    const w = C.WEAVES.find((x) => x.key === it.weave);
    const conf = { 1: 'confidence 1 low', 2: 'confidence 2 medium', 3: 'confidence 3 high' }[it.confidence] || 'confidence not set';
    const fl = it.flags.map((k) => (C.FLAGS.find((f) => f.key === k) || { label: k }).label);
    if (it.nonwoven) fl.push('not woven');
    const t3 = add('t3', '');
    t3.textContent = (w ? w.label : 'weave not set') + ' · ' + conf + (fl.length ? ' · ' + fl.join(', ') : ' · no flags');
    if (it.empty.length) add('t3', '<span class="bad">' + it.empty.map((l) => LINE_NAME[l]).join(' and ') + ' line has no dots (cleared?)</span>');
    b.onclick = () => { const id = it.id; setExamples(false); show(id); };
    return b;
  }
  function smallCrop(img) {
    const src = img.crop;
    if (ex.small.has(src)) return ex.small.get(src);
    const p = new Promise((resolve, reject) => {
      const el = new Image();
      el.onload = () => resolve(el);
      el.onerror = () => reject(new Error('failed to load ' + src));
      el.src = src;
    }).then((el) => {
      const c = document.createElement('canvas'); c.width = THUMB_PX; c.height = THUMB_PX;
      const cx = c.getContext('2d'); cx.imageSmoothingEnabled = true; cx.imageSmoothingQuality = 'high';
      cx.drawImage(el, 0, 0, THUMB_PX, THUMB_PX);
      return c;
    });
    ex.small.set(src, p);
    p.catch(() => ex.small.delete(src));
    return p;
  }
  function queueThumb(cv) {
    if (cv._queued) return;
    cv._queued = true; ex.queue.push(cv); pumpThumbs();
  }
  function pumpThumbs() {
    while (ex.busy < 2 && ex.queue.length) {
      const cv = ex.queue.shift(), it = cv._it, img = it && st.byId[it.id], gen = ex.gen;
      if (!img || !cv.isConnected) continue;
      ex.busy++;
      smallCrop(img).then((small) => { if (gen === ex.gen && cv.isConnected) drawTile(cv, it, small); }, () => { /* tile stays blank */ })
        .then(() => { ex.busy--; pumpThumbs(); });
    }
  }
  /** Thumbnail: the crop turned by the record's image turn, the window, both lines at their angles and every dot (C.thumbGeometry). */
  function drawTile(cv, it, small) {
    const rec = st.store.records[String(it.id)];
    const own = !it.iaa && rec && rec.done && Array.isArray(rec.h_dots) && rec.rect_size && C.sameAnnotator(rec.annotator, st.store.annotator);
    const T = Math.max(60, Math.round(cv.clientWidth || 176)), d = window.devicePixelRatio || 1;
    cv.width = Math.round(T * d); cv.height = Math.round(T * d);
    const c = cv.getContext('2d');
    const S = own ? rec.rect_size : THUMB_PX;
    const g = own ? C.thumbGeometry(rec, T, 4) : null;
    const v = g ? g.view : { scale: (T - 8) / S, ox: 4, oy: 4, rot: 0 };
    c.setTransform(d * v.scale, 0, 0, d * v.scale, d * v.ox, d * v.oy);
    if (v.rot) { c.translate(S / 2, S / 2); c.rotate(v.rot * Math.PI / 180); c.translate(-S / 2, -S / 2); }
    c.imageSmoothingEnabled = true; c.imageSmoothingQuality = 'high';
    if (!own) c.globalAlpha = 0.6;
    c.drawImage(small, 0, 0, S, S);
    c.globalAlpha = 1;
    c.setTransform(d, 0, 0, d, 0, 0);
    if (!g) return;   // IAA-subset scene (or no longer this annotator's done record): the photo only
    const css = getComputedStyle(document.documentElement), col = (k, f) => css.getPropertyValue(k).trim() || f;
    c.strokeStyle = 'rgba(255,255,255,0.5)'; c.lineWidth = 1; c.setLineDash([3, 3]);
    c.beginPath(); g.window.forEach((p, i) => { if (i) c.lineTo(p[0], p[1]); else c.moveTo(p[0], p[1]); }); c.closePath(); c.stroke();
    c.setLineDash([]);
    c.strokeStyle = col('--accent', '#f5c542'); c.lineWidth = 1;
    ['h', 'v'].forEach((l) => { const A = g.lines[l][0], B = g.lines[l][1]; c.beginPath(); c.moveTo(A[0], A[1]); c.lineTo(B[0], B[1]); c.stroke(); });
    [['h', col('--hdot', '#22d3ee')], ['v', col('--vdot', '#f472b6')]].forEach(([l, color]) => {
      c.fillStyle = color; c.strokeStyle = 'rgba(0,0,0,0.7)'; c.lineWidth = 0.5;
      g.dots[l].forEach((p) => { c.beginPath(); c.arc(p[0], p[1], 1.8, 0, Math.PI * 2); c.fill(); c.stroke(); });
    });
  }

  // ------------------------------------------------------------------ keyboard
  function isTyping(t) {
    if (!t) return false;
    if (t.tagName === 'TEXTAREA' || t.tagName === 'SELECT') return true;
    return t.tagName === 'INPUT' && !['checkbox', 'range', 'button', 'file'].includes(t.type);
  }
  function onKeyDown(e) {
    noteInput();
    if (e.key === 'Escape') {
      setHelp(false);
      setExamples(false);
      if (isTyping(e.target)) e.target.blur();
      return;
    }
    if (isTyping(e.target)) {
      if (e.target.id === 'jump' && e.key === 'Enter') e.target.blur(); // blur fires 'change' -> show()
      return;
    }
    const k = e.key;
    // v2.2.2: while the Examples panel is open no shortcut acts on the image; X (or Esc) closes it
    if (examplesOpen()) {
      if ((k === 'x' || k === 'X') && !e.metaKey && !e.ctrlKey && !e.altKey) { setExamples(false); e.preventDefault(); }
      return;
    }
    if ((e.metaKey || e.ctrlKey) && k.toLowerCase() === 'z') { e.preventDefault(); undo(); return; }
    if ((e.metaKey || e.ctrlKey) && k.toLowerCase() === 's') { e.preventDefault(); saveDone(false); return; }
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    if (!$('help').hidden && !['?', 'h', 'H', 'x', 'X'].includes(k)) return;
    let handled = true;
    if (e.code === 'BracketLeft' || k === '[' || k === '{') { rotateSel(e.shiftKey ? -5 : -0.5); e.preventDefault(); return; }
    if (e.code === 'BracketRight' || k === ']' || k === '}') { rotateSel(e.shiftKey ? 5 : 0.5); e.preventDefault(); return; }
    // v2.2.1: turn the image (, . = -/+0.5 deg, Shift 5 deg; / = back to 0). '?' (Shift-/) stays help.
    if (e.code === 'Comma' || k === ',' || k === '<') { rotateImage(e.shiftKey ? -5 : -0.5, 'key'); e.preventDefault(); return; }
    if (e.code === 'Period' || k === '.' || k === '>') { rotateImage(e.shiftKey ? 5 : 0.5, 'key'); e.preventDefault(); return; }
    if (k === '/') { viewZero(); e.preventDefault(); return; }
    switch (k) {
      case 'ArrowRight': step(1); break;
      case 'ArrowLeft': step(-1); break;
      case 'n': case 'N': nextTodo(); break;
      case 'g': case 'G': goToRep(); break;
      case 'b': case 'B': toggleBlind(); break;
      case 's': case 'S': saveDone(false); break;
      case 'Enter': saveDone(true); break;
      case 'z': case 'Z': undo(); break;
      case 'f': case 'F': fitView(); break;
      case 'e': case 'E': setEnhanced(!st.prefs.enhanced); break;
      case 'w': case 'W': if (st.rec) cycle('warp_dir', C.WARP_DIRS.map((x) => x.key)); break;
      case 'c': case 'C': if (st.rec) cycle('confidence', [1, 2, 3]); break;
      case 'l': case 'L': selectLine(st.prefs.selLine === 'h' ? 'v' : 'h'); toast(LINE_NAME[st.prefs.selLine] + ' line selected', 900); break;
      case 'a': case 'A': autoAlign(e.shiftKey ? ['h', 'v'] : [st.prefs.selLine]); break;
      case 'r': case 'R': alignImage(); break;
      case '0': zeroSel(); break;
      case '+': case '=': zoomAt(cssW / 2, cssH / 2, 1.25); break;
      case '-': case '_': zoomAt(cssW / 2, cssH / 2, 0.8); break;
      case '?': case 'h': case 'H': setHelp($('help').hidden); break;
      case 'x': case 'X': setExamples(true); break;
      case ' ':
        st.spaceDown = true; stage.classList.add('space'); break;
      default:
        if (/^[1-7]$/.test(k)) setWeave(C.WEAVES[Number(k) - 1].key);
        else handled = false;
    }
    if (handled) e.preventDefault();
  }
  function onKeyUp(e) {
    if (e.key === ' ') { st.spaceDown = false; stage.classList.remove('space'); }
  }

  function setEnhanced(on) {
    st.prefs.enhanced = !!on; savePrefs();
    $('enhanced').checked = st.prefs.enhanced;
    if (st.cur) refreshImage(false);
  }

  // ------------------------------------------------------------------ wiring
  function wire() {
    stage.addEventListener('pointerdown', onPointerDown);
    stage.addEventListener('pointermove', onPointerMove);
    stage.addEventListener('pointerup', onPointerUp);
    stage.addEventListener('pointercancel', onPointerUp);
    stage.addEventListener('pointerleave', () => { if (!st.drag && st.hover) { st.hover = null; requestDraw(); } });
    stage.addEventListener('wheel', onWheel, { passive: false });
    stage.addEventListener('mousedown', (e) => { if (e.button === 1) e.preventDefault(); });
    stage.addEventListener('auxclick', (e) => e.preventDefault());
    stage.addEventListener('contextmenu', (e) => e.preventDefault());
    // any input anywhere counts as activity (capture phase: before the handlers act on it)
    document.addEventListener('pointerdown', noteInput, true);
    document.addEventListener('wheel', noteInput, { capture: true, passive: true });
    document.addEventListener('keydown', onKeyDown);
    document.addEventListener('keyup', onKeyUp);
    window.addEventListener('blur', () => { st.spaceDown = false; stage.classList.remove('space'); });
    if (window.ResizeObserver) new ResizeObserver(() => resizeCanvases()).observe(stage);
    else window.addEventListener('resize', () => { resizeCanvases(); });
    // keep keyboard shortcuts working after clicking buttons / checkboxes / sliders
    document.addEventListener('click', (e) => {
      const t = e.target.closest('button, input[type=checkbox], input[type=range]');
      if (t && !t.closest('.modal')) setTimeout(() => t.blur(), 0);
    });
    window.addEventListener('beforeunload', () => { if (st.rec) { flushTime(); saveNow(); } });
    // another tab of the same annotator saved: merge (newest record wins) so tabs never clobber each other
    window.addEventListener('storage', (e) => {
      if (e.key === storeKey(st.store.annotator) && e.newValue) {
        try { if (mergeStored(JSON.parse(e.newValue))) renderSide(); } catch (err) { /* ignore */ }
      }
    });
    document.addEventListener('visibilitychange', () => {
      if (document.hidden && st.rec) { flushTime(); st.lastInput = Date.now(); saveNow(); }
      updateTimer();
    });

    $('scenego').onclick = goToRep;
    $('sceneunlock').onclick = unlockCopy;
    $('prev').onclick = () => step(-1);
    $('next').onclick = () => step(1);
    $('jump').addEventListener('change', (e) => { if (e.target.value !== '') show(Number(e.target.value)); });
    $('filter').onchange = (e) => { st.prefs.filter = e.target.value; savePrefs(); renderSide(); e.target.blur(); };
    $('annotator').addEventListener('change', (e) => switchAnnotator(e.target.value));
    $('annotator').addEventListener('keydown', (e) => { if (e.key === 'Enter') e.target.blur(); });
    $('download').onclick = exportAll;
    $('importbtn').onclick = () => $('importfile').click();
    $('importfile').onchange = (e) => { const f = e.target.files[0]; if (f) importFile(f); e.target.value = ''; };
    $('helpbtn').onclick = () => setHelp(true);
    $('helpclose').onclick = () => setHelp(false);
    $('help').addEventListener('click', (e) => { if (e.target.id === 'help') setHelp(false); });
    $('abouthelp').onclick = (e) => { e.preventDefault(); setHelp(true); };
    // v2.2.2: Examples panel
    $('examplesbtn').onclick = () => setExamples(true);
    $('exclose').onclick = () => setExamples(false);
    $('extab-look').onclick = () => exTab('look');
    $('extab-mine').onclick = () => exTab('mine');
    $('examples').addEventListener('click', (e) => { if (e.target.id === 'examples') setExamples(false); });
    $('examples').addEventListener('keydown', (e) => {   // arrow keys switch tabs when a tab has the focus
      if ((e.key === 'ArrowLeft' || e.key === 'ArrowRight') && e.target && e.target.getAttribute('role') === 'tab') {
        exTab(ex.tab === 'look' ? 'mine' : 'look'); e.preventDefault();
      }
    });

    // guide-line card
    $('selh').onclick = () => selectLine('h');
    $('selv').onclick = () => selectLine('v');
    $('counth').onclick = () => selectLine('h');
    $('countv').onclick = () => selectLine('v');
    $('rotm').onclick = (e) => rotateSel(e.shiftKey ? -5 : -0.5);
    $('rotp').onclick = (e) => rotateSel(e.shiftKey ? 5 : 0.5);
    $('rot0').onclick = zeroSel;
    $('autoalign').onclick = (e) => autoAlign(e.shiftKey ? ['h', 'v'] : [st.prefs.selLine]);
    $('autoalign').addEventListener('mouseenter', () => {
      const est = estNow(st.cur), l = st.prefs.selLine;
      if (est && est[l]) { st.preview = { line: l, deg: est[l].deg }; requestDraw(); }
    });
    $('autoalign').addEventListener('mouseleave', () => { if (st.preview) { st.preview = null; requestDraw(); } });
    // v2.2.1: image turn
    $('viewrot').addEventListener('input', (e) => {
      if (!st.rec || readOnly()) { renderLines(); return; }
      const want = Math.round(Number(e.target.value) * 100) / 100;
      if (isFinite(want)) rotateImage(want - C.viewAngle(st.rec), 'slider');
    });
    $('viewrot').addEventListener('change', (e) => { e.target.blur(); renderLines(); });
    $('viewm').onclick = (e) => rotateImage(e.shiftKey ? -5 : -0.5, 'key');
    $('viewp').onclick = (e) => rotateImage(e.shiftKey ? 5 : 0.5, 'key');
    $('view0').onclick = viewZero;
    $('viewalign').onclick = alignImage;
    $('removeout').onclick = () => {
      if (!st.rec || locked()) return;
      pushUndo();
      const n = C.removeOutside(st.rec, 'h') + C.removeOutside(st.rec, 'v');
      if (n) commit('Removed ' + n + ' dot' + (n > 1 ? 's' : '') + ' outside the window (Z to undo)'); else popUndoIfUnchanged(0);
    };

    $('blind').onchange = () => toggleBlind();
    $('blindpolicy').onchange = (e) => {
      st.prefs.blindPolicy = e.target.value; savePrefs(); e.target.blur(); renderSide();
      toast('Applies to images you have not opened yet (an image is "opened" the first time it is shown)');
    };
    $('nonwoven').onchange = (e) => setField('nonwoven', e.target.checked);
    let notesTimer = null;
    $('notes').addEventListener('input', (e) => {
      if (!st.rec || readOnly()) return;
      st.rec.notes = e.target.value; C.touch(st.rec);
      clearTimeout(notesTimer); notesTimer = setTimeout(() => { scheduleSave(); }, 400);
    });
    $('notes').addEventListener('blur', () => { if (st.rec && !readOnly()) commit(); });
    $('savebtn').onclick = () => saveDone(false);
    $('undodone').onclick = () => { if (st.rec && st.rec.done && !locked()) setField('done', false); };
    $('undo').onclick = undo;
    $('reset').onclick = () => {
      if (!st.rec || locked()) return;
      if (st.rec.blind_mode) { toast('Blind mode is on: turn it off first to see proposals'); return; }
      if (!window.confirm('Replace all dots on both lines of image ' + st.rec.id + ' with the proposals (lines back to 0°)?')) return;
      pushUndo(); C.resetToProposals(st.rec, st.cur); commit('Reset to proposals');
    };
    $('clearh').onclick = () => { if (st.rec && !locked()) { pushUndo(); C.clearLine(st.rec, 'h'); commit('Cleared H dots (Z to undo)'); } };
    $('clearv').onclick = () => { if (st.rec && !locked()) { pushUndo(); C.clearLine(st.rec, 'v'); commit('Cleared V dots (Z to undo)'); } };
    $('enhanced').onchange = (e) => setEnhanced(e.target.checked);
    $('fit').onclick = fitView;
    [['brightness', 'brightness'], ['contrast', 'contrast'], ['dotsize', 'dotSize']].forEach(([id, key]) => {
      $(id).addEventListener('input', (e) => { st.prefs[key] = Number(e.target.value); savePrefs(); requestDraw(); });
    });
    $('showgaps').onchange = (e) => { st.prefs.showGaps = e.target.checked; savePrefs(); requestDraw(); };

    // active time on image (see commitPending) + periodic save of the credited seconds
    setInterval(tickTime, 1000);
    setInterval(() => { if (st.timeDirty) saveNow(); }, 15000);
  }

  // ------------------------------------------------------------------ boot
  async function boot() {
    loadPrefs();
    buildSide();
    wire();
    resizeCanvases();
    $('appversion').textContent = C.APP_VERSION;
    $('aboutversion').textContent = C.APP_VERSION + ' (' + C.TOOL_VERSION + ')';
    $('annotator').value = st.prefs.annotator;
    $('filter').value = st.prefs.filter;
    $('enhanced').checked = st.prefs.enhanced;
    $('brightness').value = st.prefs.brightness;
    $('contrast').value = st.prefs.contrast;
    $('dotsize').value = st.prefs.dotSize;
    $('showgaps').checked = st.prefs.showGaps;
    $('blindpolicy').value = st.prefs.blindPolicy;

    try {
      const r = await fetch('data/manifest.json', { cache: 'no-store' });
      if (!r.ok) throw new Error('HTTP ' + r.status);
      st.manifest = await r.json();
    } catch (e) {
      $('loading').hidden = false;
      $('loading').innerHTML = 'Could not load <code>data/manifest.json</code> (' + e.message + ').<br>' +
        'Run <code>python prepare.py</code>, then serve this folder with <code>python3 server.py</code>.';
      return;
    }
    st.images = st.manifest.images.slice().sort((a, b) => a.id - b.id);
    st.images.forEach((im, i) => { st.byId[im.id] = im; st.order[im.id] = i; });
    st.iaa = new Set(st.manifest.iaa_subset || []);
    $('jump').max = st.images[st.images.length - 1].id;
    // v2.1: scene-aware when the manifest carries scene fields (prepare.py --scenes-only)
    st.scenes = C.hasScenes(st.manifest);
    const scOpt = $('filter').querySelector('option[value="scenes"]');
    if (st.scenes) {
      scOpt.textContent = 'Scenes to annotate (' + st.images.filter((im) => !C.isCopy(im)).length + ')';
    } else {
      scOpt.remove();
      if (st.prefs.filter === 'scenes') st.prefs.filter = 'all';
    }
    // v2.2: "High attention first" only with a manifest that has attention fields (and a matching core.js)
    st.attention = typeof C.hasAttention === 'function' && C.hasAttention(st.manifest);
    if (!st.attention) {
      const ao = $('attnopt'); if (ao) ao.remove();
      if (st.prefs.filter === 'attention') st.prefs.filter = st.scenes ? 'scenes' : 'all';
    }
    $('filter').value = st.prefs.filter;
    $('iaaopt').textContent = 'IAA subset (' + st.iaa.size + ')';

    st.serverMode = await detectServer();
    st.store = loadLocalStore(st.prefs.annotator);
    st.store.annotator = st.prefs.annotator;
    if (st.serverMode) {
      const srv = await loadServerStore(st.prefs.annotator);
      if (srv) {
        const m = C.mergeRecords(st.store.records, srv.records);
        st.store.records = m.merged;
        if (m.added || m.updated) toast('Restored ' + (m.added + m.updated) + ' records from the disk autosave', 3000);
      }
    }
    updateSaveStatus();

    let start = st.prefs.lastId != null && st.byId[st.prefs.lastId] ? st.prefs.lastId : st.images[0].id;
    let redirect = null;
    if (st.scenes && C.isCopy(st.byId[start]) && filteredIds().indexOf(start) < 0) {
      const r = C.sceneRepId(st.byId[start]);
      redirect = 'Image ' + start + ' is another file of scene #' + r + ' (labels are copied): showing #' + r +
        ', the photo to annotate. Work saved on #' + start + ' is kept.';
      start = r;
    }
    await show(start);
    let delay = 0;
    if (!st.prefs.annotator) {
      toast('Type your name in "Annotator" first (needed for inter-annotator agreement)', 5000);
      $('annotator').focus();
      delay = 5200;
    }
    if (redirect) setTimeout(() => { if (st.cur && st.cur.id === start) toast(redirect, 7000); }, delay);
  }

  window.addEventListener('error', (e) => toast('Error: ' + e.message, 5000));
  boot();
  // exposed for debugging / automated checks only
  window.__annState = st;
  window.__annDebug = { getEstimate: getEstimate, show: show, autoAlign: autoAlign, lineHit: lineHit, handles: handles,
    toScreen: toScreen, toWorld: toWorld, rotateImage: rotateImage, alignImage: alignImage, viewOf: viewOf, dotHit: dotHit,
    setExamples: setExamples, exTab: exTab, examples: ex };
})();
