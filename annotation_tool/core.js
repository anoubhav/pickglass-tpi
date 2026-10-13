/*
 * core.js — pure annotation logic for the thread-density annotator (no DOM).
 * Loaded by index.html (as window.AnnCore) and by node tests (require('./core.js')).
 *
 * Coordinates are RECTIFIED crop pixels (see common/rectify.py): the crop is S x S and
 * covers exactly the 1-inch pick-glass window.
 *   H line = guide through the pivot (S/2, h_line_y); its dots are x positions where
 *            VERTICAL yarns cross it  -> "H-line count (vertical yarns)".
 *   V line = guide through the pivot (v_line_x, S/2); its dots are y positions where
 *            HORIZONTAL yarns cross it  -> "V-line count (horizontal yarns)".
 *
 * v2: each line may be rotated about its pivot by `<line>_angle_deg` (absent = 0, the v1
 * axis-aligned behaviour; |angle| <= 45). Image coordinates have y pointing down, so a
 * positive angle turns a line CLOCKWISE on screen; the H line direction is
 * (cos a, sin a) and the V line direction is (-sin a, cos a), so equal angles keep the
 * two lines perpendicular. Dots keep their parameter (x for H, y for V) and always sit
 * on the line: H dot x -> (x, h_line_y + (x - S/2) tan a). The part of the line inside the
 * window is the SEGMENT; threads/inch for a line = (dots on the segment) / (segment
 * length in inches). At angle 0 the segment is exactly 1 inch, so TPI == dot count.
 *
 * v2.2.1: the IMAGE can be turned on screen by `view_rot_deg` (absent = 0, |view| <= 45, same sign
 * convention: + = clockwise) about the crop centre (S/2, S/2). The turn is RIGID with the guide
 * lines: turning the view by d also adds -d to both line angles, so the lines keep their angle on
 * screen while the fabric turns under them. Counting is unchanged: dots stay in image coordinates
 * and TPI is the same in-window line-length normalisation (a view turn by d is exactly a rotation
 * of both lines by -d). viewToScreen / screenToWorld map image <-> screen (pan, zoom, rotation).
 */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) { module.exports = factory(); return; }
  root.AnnCore = factory();
  // Web Worker entry (app.js: new Worker('core.js')): orientation estimates off the UI thread.
  // message {id, gray, N, opts} -> {id, result} | {id, error}
  if (typeof window === 'undefined' && root && typeof root.postMessage === 'function' && typeof importScripts === 'function') {
    root.onmessage = function (e) {
      const d = e.data || {};
      try { root.postMessage({ id: d.id, result: root.AnnCore.estimateOrientation(d.gray, d.N, d.opts) }); }
      catch (err) { root.postMessage({ id: d.id, error: String(err && err.message || err) }); }
    };
  }
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  const SCHEMA = 1;              // record layout is unchanged; v2 only ADDS optional fields
  // 2.0.1: undo keeps the audit trail; rotation-limit undo fix
  // 2.1.0: scene-aware (manifest scene fields): reps are annotated, other files of a scene are
  //        read-only copies whose labels are copied; record format unchanged
  // 2.2.0: "check carefully" attention badge + "High attention first" filter (manifest attention
  //        fields, attention.py); never on IAA-subset scenes or blind records; record format unchanged
  // 2.2.1: the image can be turned on screen (view_rot_deg, rigid with the guide lines); optional
  //        record field view_rot_deg, CSV column view_rot_deg appended at the end. Kept at 2.2.x on
  //        purpose: paper/tools/label_day.py finds records that could show the v2.2 attention badge
  //        with tool_version.startswith("2.2"), and this version shows it exactly as 2.2.0 does.
  // 2.2.2: Examples panel (key X): example figures of non-IAA scenes (guide/examples/, make_tool_examples.py)
  //        and the current annotator's own finished scenes; record format unchanged (still 2.2.x, as above)
  const TOOL_VERSION = '2.2.2';
  const APP_VERSION = 'v2';
  const MAX_ANGLE = 45;          // degrees; a guide line never turns further than this
  const MAX_VIEW = 45;           // degrees; the image (view) never turns further than this
  const ANGLE_LOG_MAX = 200;
  const D2R = Math.PI / 180;
  const WEAVES = [
    { key: 'plain', label: 'Plain 1/1' },
    { key: 'twill_2_1', label: 'Twill 2/1' },
    { key: 'twill_3_1', label: 'Twill 3/1' },
    { key: 'twill_2_2', label: 'Twill 2/2' },
    { key: 'satin', label: 'Satin' },
    { key: 'other', label: 'Other' },
    { key: 'unsure', label: 'Unsure' },
  ];
  const WARP_DIRS = [
    { key: 'vertical', label: 'Vertical' },
    { key: 'horizontal', label: 'Horizontal' },
    { key: 'unknown', label: 'Unknown' },
  ];
  const FLAGS = [
    { key: 'glare', label: 'Glare' },
    { key: 'blur', label: 'Blur' },
    { key: 'shadow', label: 'Shadow' },
    { key: 'low_res', label: 'Low-res' },
    { key: 'perspective', label: 'Perspective' },
    { key: 'occluded', label: 'Partially occluded' },
  ];

  const nz = (v) => (v === 0 ? 0 : v);   // -0 -> 0 (JSON and deepStrictEqual friendly)
  const r1 = (v) => nz(Math.round(v * 10) / 10);
  const r2 = (v) => nz(Math.round(v * 100) / 100);
  const r4 = (v) => nz(Math.round(v * 10000) / 10000);
  const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
  const nowIso = () => new Date().toISOString();

  function emptyFlags() {
    const f = {};
    FLAGS.forEach((x) => { f[x.key] = false; });
    return f;
  }

  /**
   * Identity of the proposal an image currently has in the manifest (method + FNV-1a hash
   * of the guide-line positions). Stored in each record so the app can tell when
   * `prepare.py --refresh-proposals` has produced newer proposals.
   */
  function proposalSig(img) {
    const p = (img && img.proposal) || {};
    const s = [p.method || '', p.h_line_y, p.v_line_x,
      (p.h_positions || []).join(','), (p.v_positions || []).join(',')].join('|');
    let h = 2166136261 >>> 0;
    for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619) >>> 0; }
    return (p.method || 'none') + ':' + h.toString(16);
  }

  /** Deterministic 50% assignment used by the "random 50%" blind policy. */
  function hashBlind(id) {
    let h = (Number(id) + 1) * 2654435761;
    h = (h ^ (h >>> 15)) >>> 0;
    return (h & 1) === 1;
  }

  /**
   * Blind setting for an image opened for the first time.
   *   'iaa'    blind iff the image is in the inter-annotator-agreement subset (default)
   *   'random' deterministic 50% by id;  'all' / 'none';  'manual' = follow the B toggle
   */
  function blindForPolicy(policy, id, manualDefault, inIaa) {
    if (policy === 'iaa') return !!inIaa;
    if (policy === 'all') return true;
    if (policy === 'none') return false;
    if (policy === 'random') return hashBlind(id);
    return !!manualDefault; // 'manual'
  }

  // ------------------------------------------------------------------ line geometry (v2)
  /** Angle of `line` in degrees (missing / invalid -> 0, clamped to +-MAX_ANGLE). */
  function lineAngle(rec, line) {
    const a = Number(rec[line + '_angle_deg']);
    return isFinite(a) ? nz(clamp(a, -MAX_ANGLE, MAX_ANGLE)) : 0;
  }
  /** Rotation centre of the line: (S/2, h_line_y) for H, (v_line_x, S/2) for V. */
  function linePivot(rec, line) {
    const S = rec.rect_size;
    return line === 'h' ? [S / 2, rec.h_line_y] : [rec.v_line_x, S / 2];
  }
  /** Unit direction of the line (y down): H (cos a, sin a), V (-sin a, cos a). */
  function lineDir(rec, line) {
    const a = lineAngle(rec, line);
    if (a === 0) return line === 'h' ? [1, 0] : [0, 1];
    const t = a * D2R;
    return line === 'h' ? [Math.cos(t), Math.sin(t)] : [-Math.sin(t), Math.cos(t)];
  }
  /** World (x, y) of parameter p on the line (H: p = x, V: p = y). */
  function pointAt(rec, line, p) {
    const a = lineAngle(rec, line), S = rec.rect_size;
    if (line === 'h') return a === 0 ? [p, rec.h_line_y] : [p, rec.h_line_y + (p - S / 2) * Math.tan(a * D2R)];
    return a === 0 ? [rec.v_line_x, p] : [rec.v_line_x - (p - S / 2) * Math.tan(a * D2R), p];
  }
  /** Parameter interval {lo, hi} of the part of the line inside the S x S window. */
  function paramRange(rec, line) {
    const S = rec.rect_size, a = lineAngle(rec, line);
    if (a === 0) return { lo: 0, hi: S };
    // the other coordinate along the line: c(p) = c0 + (p - S/2) k, must stay in [0, S]
    const k = (line === 'h' ? 1 : -1) * Math.tan(a * D2R);
    const c0 = line === 'h' ? rec.h_line_y : rec.v_line_x;
    let lo = S / 2 + (0 - c0) / k, hi = S / 2 + (S - c0) / k;
    if (lo > hi) { const t = lo; lo = hi; hi = t; }
    lo = Math.max(0, lo); hi = Math.min(S, hi);
    if (hi < lo) hi = lo;
    return { lo: lo, hi: hi };
  }
  /** End points [[x0, y0], [x1, y1]] of the segment inside the window (lo end first). */
  function segment(rec, line) {
    const r = paramRange(rec, line);
    return [pointAt(rec, line, r.lo), pointAt(rec, line, r.hi)];
  }
  /**
   * Orthogonal projection of world point (wx, wy) onto the line.
   * -> {pos: parameter clamped to the segment, raw: unclamped parameter,
   *     dist: perpendicular distance (world px), inside: raw within the segment}
   */
  function projectToLine(rec, line, wx, wy) {
    const pv = linePivot(rec, line), u = lineDir(rec, line);
    const dx = wx - pv[0], dy = wy - pv[1];
    const t = dx * u[0] + dy * u[1];
    const dist = Math.abs(dx * u[1] - dy * u[0]);
    const p = line === 'h' ? pv[0] + t * u[0] : pv[1] + t * u[1];
    const r = paramRange(rec, line);
    return { pos: clamp(p, r.lo, r.hi), raw: p, dist: dist, inside: p >= r.lo && p <= r.hi };
  }
  /**
   * Everything about one line: angle, segment (param range, length in px and inches),
   * dot counts and the normalised threads per inch = dots on the segment / length (in).
   * k = segment length per unit of parameter (1 / cos a). At angle 0: length = 1 inch
   * exactly and tpi == number of dots (v1 identity).
   */
  function lineInfo(rec, line) {
    const S = rec.rect_size, a = lineAngle(rec, line), r = paramRange(rec, line);
    const cos = a === 0 ? 1 : Math.cos(a * D2R);
    const segPx = (r.hi - r.lo) / cos;
    const segIn = segPx / S;
    const dots = rec[line + '_dots'] || [];
    let nIn = dots.length;
    if (a !== 0) nIn = dots.filter((p) => p >= r.lo - 0.5 && p <= r.hi + 0.5).length;
    const tpi = segIn > 0 ? (a === 0 ? nIn : nIn / segIn) : null;
    return { angle: a, lo: r.lo, hi: r.hi, k: 1 / cos, segPx: segPx, segIn: segIn,
      n: dots.length, nIn: nIn, nOut: dots.length - nIn, tpi: tpi };
  }
  /** How the line's angle was last set: 'auto' | 'key' | 'drag' | 'zero' | 'reset' | 'undo' | ''. */
  function angleSource(rec, line) {
    const log = Array.isArray(rec.angle_log) ? rec.angle_log : [];
    for (let i = log.length - 1; i >= 0; i--) if (log[i] && log[i].line === line) return log[i].how || '';
    return '';
  }
  /** Derived per-line fields stored in v2 records and exports (always recomputable). */
  function derived(rec) {
    const o = {};
    ['h', 'v'].forEach((l) => {
      const L = lineInfo(rec, l);
      o[l + '_angle_deg'] = r2(L.angle);
      o[l + '_seg_len_px'] = r1(L.segPx);
      o[l + '_seg_len_in'] = r4(L.segIn);
      o[l + '_raw_count'] = L.n;
      o[l + '_tpi'] = L.tpi == null ? null : r2(L.tpi);
    });
    return o;
  }
  function updateDerived(rec) {
    if (rec && Array.isArray(rec.h_dots) && Array.isArray(rec.v_dots) && rec.rect_size) Object.assign(rec, derived(rec));
    return rec;
  }
  function withDerived(rec) { return Object.assign({}, rec, derived(rec)); }

  function logAngle(rec, line, deg, how, extra) {
    if (!Array.isArray(rec.angle_log)) rec.angle_log = [];
    const log = rec.angle_log, now = nowIso(), last = log[log.length - 1];
    // a burst of key presses / one drag is a single entry
    if (last && last.line === line && last.how === how && (how === 'key' || how === 'drag') &&
        Date.parse(now) - Date.parse(last.t) < 3000) {
      last.t = now; last.deg = deg; return;
    }
    log.push(Object.assign({ t: now, line: line, deg: deg, how: how || 'manual' }, extra || {}));
    if (log.length > ANGLE_LOG_MAX) log.splice(0, log.length - ANGLE_LOG_MAX);
  }
  /**
   * Rotate `line` to `deg` (clamped to +-45, 0.01 deg) about its pivot. Dots keep their
   * parameter and therefore stay on the line. Logged in rec.angle_log with `how`.
   * Returns false (and does not touch the record) when nothing changes.
   */
  function setAngle(rec, line, deg, how, extra) {
    let a = Number(deg);
    if (!isFinite(a)) a = 0;
    a = r2(clamp(a, -MAX_ANGLE, MAX_ANGLE));
    if (a === lineAngle(rec, line)) return false;
    rec[line + '_angle_deg'] = a;
    logAngle(rec, line, a, how, extra);
    touch(rec);
    return true;
  }
  function rotateBy(rec, line, delta, how) {
    return setAngle(rec, line, lineAngle(rec, line) + delta, how || 'key');
  }

  // ------------------------------------------------------------------ image (view) rotation (v2.2.1)
  /** Turn of the image on screen in degrees (y down, + = clockwise; missing / invalid -> 0, clamped to +-MAX_VIEW). */
  function viewAngle(rec) {
    const a = Number(rec && rec.view_rot_deg);
    return isFinite(a) ? nz(clamp(a, -MAX_VIEW, MAX_VIEW)) : 0;
  }
  /**
   * Range {lo, hi} of view turns d that are possible right now: the view stays within +-MAX_VIEW
   * and both lines (which turn by -d) within +-MAX_ANGLE in the image.
   */
  function viewRange(rec) {
    const v = viewAngle(rec), ah = lineAngle(rec, 'h'), av = lineAngle(rec, 'v');
    return { lo: r2(Math.max(-MAX_VIEW - v, ah - MAX_ANGLE, av - MAX_ANGLE)),
      hi: r2(Math.min(MAX_VIEW - v, ah + MAX_ANGLE, av + MAX_ANGLE)) };
  }
  /** angle_log entries of a view turn: one per line, how 'view', with the new view angle and its source. */
  function logView(rec, src, extra) {
    if (!Array.isArray(rec.angle_log)) rec.angle_log = [];
    const log = rec.angle_log, now = nowIso(), v = viewAngle(rec);
    const a = log[log.length - 2], b = log[log.length - 1];
    const burst = (e) => e && e.how === 'view' && e.src === src && e.est == null;
    // a burst of key presses / one slider drag is a single pair of entries
    if ((src === 'key' || src === 'slider') && !extra && burst(a) && burst(b) && a.line !== b.line &&
        Date.parse(now) - Date.parse(b.t) < 3000) {
      [a, b].forEach((e) => { e.t = now; e.deg = lineAngle(rec, e.line); e.view = v; });
      return;
    }
    ['h', 'v'].forEach((l) => log.push(Object.assign({ t: now, line: l, deg: lineAngle(rec, l), how: 'view', view: v, src: src || 'key' }, extra || {})));
    if (log.length > ANGLE_LOG_MAX) log.splice(0, log.length - ANGLE_LOG_MAX);
  }
  /**
   * Turn the image by `delta` degrees on screen, rigidly with the guide lines: view_rot_deg += d and
   * both line angles -= d (so each line keeps its angle on screen and its pivot, a fabric point).
   * d is rounded to 0.01 deg and clamped to viewRange(). TPI is then exactly what rotating both lines
   * by -d gives (same lineInfo). Logged in angle_log (how 'view', src e.g. key | slider | zero | align).
   * Returns the turn applied (0: nothing changed, the record is not touched).
   */
  function rotateView(rec, delta, src, extra) {
    let d = Number(delta);
    if (!isFinite(d)) return 0;
    const rg = viewRange(rec), v = viewAngle(rec), ah = lineAngle(rec, 'h'), av = lineAngle(rec, 'v');
    d = r2(clamp(r2(d), rg.lo, rg.hi));
    if (!d) return 0;
    rec.view_rot_deg = r2(v + d);
    rec.h_angle_deg = r2(clamp(ah - d, -MAX_ANGLE, MAX_ANGLE));
    rec.v_angle_deg = r2(clamp(av - d, -MAX_ANGLE, MAX_ANGLE));
    logView(rec, src, extra);
    touch(rec);
    return d;
  }
  /** Turn the image to `deg` (rigidly, as rotateView). Returns the turn applied. */
  function setViewAngle(rec, deg, src, extra) {
    const a = Number(deg);
    return isFinite(a) ? rotateView(rec, r2(a) - viewAngle(rec), src, extra) : 0;
  }
  /**
   * "Rotate image to align": the view turn after which `line` lies at the estimated yarn angle
   * `estDeg` (image convention) while keeping its angle on screen, i.e. the fabric is turned until its
   * yarns cross the line at right angles. A level line on screen therefore ends level with the yarns
   * axis-aligned on screen. d = lineAngle - estDeg.
   */
  function viewAlignDelta(rec, line, estDeg) {
    const e = Number(estDeg);
    return isFinite(e) ? r2(lineAngle(rec, line) - e) : 0;
  }
  /**
   * Image -> screen: o + scale * (c + R(rot) (w - c)), c = (S/2, S/2); view = {scale, ox, oy, rot (deg)}.
   * At rot 0 this is exactly the v2.2.0 mapping o + scale * w.
   */
  function viewToScreen(view, S, wx, wy) {
    const t = (Number(view.rot) || 0) * D2R;
    if (!t) return [view.ox + view.scale * wx, view.oy + view.scale * wy];
    const c = S / 2, co = Math.cos(t), si = Math.sin(t), dx = wx - c, dy = wy - c;
    return [view.ox + view.scale * (c + co * dx - si * dy), view.oy + view.scale * (c + si * dx + co * dy)];
  }
  /** Screen -> image: the exact inverse of viewToScreen (pointer mapping for clicks, drags and hit tests). */
  function screenToWorld(view, S, px, py) {
    const t = (Number(view.rot) || 0) * D2R;
    const qx = (px - view.ox) / view.scale, qy = (py - view.oy) / view.scale;
    if (!t) return [qx, qy];
    const c = S / 2, co = Math.cos(t), si = Math.sin(t), rx = qx - c, ry = qy - c;
    return [c + co * rx + si * ry, c - si * rx + co * ry];
  }
  /** Screen direction of an image-space direction u (the view turn only). */
  function viewDir(view, u) {
    const t = (Number(view.rot) || 0) * D2R;
    if (!t) return [u[0], u[1]];
    const co = Math.cos(t), si = Math.sin(t);
    return [co * u[0] - si * u[1], si * u[0] + co * u[1]];
  }
  /** Width (= height) of the screen bounding box of the turned S x S crop, per unit of S. */
  function viewExtent(rot) {
    const t = (Number(rot) || 0) * D2R;
    return t ? Math.abs(Math.cos(t)) + Math.abs(Math.sin(t)) : 1;
  }
  /**
   * Pivot coordinate (H: h_line_y, V: v_line_x) at which the line, at its current angle, passes through
   * image point (wx, wy). Dragging a line by pivotThrough(pointer) + offset keeps the grabbed point under
   * the pointer for any line angle and view turn (at angle 0: H -> wy, V -> wx, as in v2.2.0).
   */
  function pivotThrough(rec, line, wx, wy) {
    const a = lineAngle(rec, line), S = rec.rect_size;
    if (a === 0) return line === 'h' ? wy : wx;
    const t = Math.tan(a * D2R);
    return line === 'h' ? wy - (wx - S / 2) * t : wx + (wy - S / 2) * t;
  }

  /**
   * Undo restores an older snapshot of a record, but the audit trail must survive it:
   * proposals_seen stays true once proposals were on screen, active time is never lost, and
   * blind_log / angle_log are append-only (a blind or angle change made by the undo itself is
   * appended, marked as such). Copies these from `current` into `restored`; returns `restored`.
   */
  function keepAudit(restored, current) {
    if (!restored || !current) return restored;
    const now = nowIso();
    restored.proposals_seen = !!(restored.proposals_seen || current.proposals_seen);
    restored.time_spent_s = Math.max(Number(restored.time_spent_s) || 0, Number(current.time_spent_s) || 0);
    if (Array.isArray(current.blind_log)) {
      restored.blind_log = current.blind_log.slice();
      if (!!restored.blind_mode !== !!current.blind_mode) restored.blind_log.push({ t: now, on: !!restored.blind_mode, undo: true });
    }
    if (Array.isArray(current.angle_log)) {
      restored.angle_log = current.angle_log.slice();
      const vr = viewAngle(restored), vturn = vr !== viewAngle(current);   // v2.2.1: an undone view turn is logged too
      ['h', 'v'].forEach((l) => {
        const a = lineAngle(restored, l);
        if (a !== lineAngle(current, l) || vturn) restored.angle_log.push(Object.assign({ t: now, line: l, deg: a, how: 'undo' }, vturn ? { view: vr } : {}));
      });
      if (restored.angle_log.length > ANGLE_LOG_MAX) restored.angle_log.splice(0, restored.angle_log.length - ANGLE_LOG_MAX);
    }
    return restored;
  }

  /** New record for manifest entry `img`. Proposals are inserted unless blind. */
  function newRecord(img, opts) {
    opts = opts || {};
    const S = img.rect_size[0];
    const p = img.proposal || {};
    const rec = {
      schema: SCHEMA,
      id: img.id,
      annotator: opts.annotator || '',
      rect_size: S,
      h_line_y: r1(clamp(p.h_line_y != null ? p.h_line_y : S / 2, 0, S)),
      v_line_x: r1(clamp(p.v_line_x != null ? p.v_line_x : S / 2, 0, S)),
      h_dots: [], h_orig: [], v_dots: [], v_orig: [],
      h_angle_deg: 0, v_angle_deg: 0,   // v2: guide-line rotation (never set automatically)
      angle_log: [],                    // v2: [{t, line, deg, how}] every rotation
      blind_mode: !!opts.blind,
      blind_initial: !!opts.blind,   // blind state when the image was first opened
      blind_log: [],                 // [{t, on}] every later blind toggle (anchoring audit)
      proposals_seen: false,         // true once machine proposals were ever on screen
      iaa: !!opts.iaa,               // in the 20-image double-annotation subset
      proposal_method: p.method || null,
      proposal_sig: proposalSig(img),
      n_prop_h: 0, n_prop_v: 0,
      weave: null, warp_dir: null, flags: emptyFlags(), nonwoven: false,
      confidence: null, notes: '',
      done: false, time_spent_s: 0, n_edits: 0, tool_version: TOOL_VERSION,
      created: opts.now || nowIso(),
      timestamp: opts.now || nowIso(),
    };
    if (!rec.blind_mode) insertProposals(rec, img);
    updateDerived(rec);
    return rec;
  }

  function sortLine(rec, line) {
    const d = rec[line + '_dots'], o = rec[line + '_orig'];
    const idx = d.map((_, i) => i).sort((a, b) => d[a] - d[b]);
    rec[line + '_dots'] = idx.map((i) => d[i]);
    rec[line + '_orig'] = idx.map((i) => o[i]);
  }

  /**
   * Add proposal dots (origin=1), skipping any closer than minSep px to an existing dot.
   * minSep == null -> automatic: 40% of the median spacing of the dots already on that
   * line (>= 1.5 px), so un-blinding after counting does not create near-duplicates.
   * Proposals that fall outside the segment of a rotated line are skipped.
   */
  function insertProposals(rec, img, minSep) {
    const p = img.proposal || {};
    const S = rec.rect_size;
    [['h', p.h_positions || []], ['v', p.v_positions || []]].forEach(([line, pos]) => {
      let sep = minSep;
      if (sep == null) {
        const g = gapStats(rec[line + '_dots'], S);
        sep = g ? Math.max(1.5, 0.4 * g.median) : 1.5;
      }
      const rg = paramRange(rec, line);
      let n = 0;
      const existing = rec[line + '_dots'].slice();
      pos.forEach((x) => {
        x = r1(clamp(Number(x), 0, S));
        if (!isFinite(x)) return;
        if (x < rg.lo - 0.05 || x > rg.hi + 0.05) return;  // off the (rotated) segment
        if (existing.some((y) => Math.abs(y - x) < sep)) return;
        if (rec[line + '_dots'].some((y) => Math.abs(y - x) < 0.05)) return; // exact duplicate
        rec[line + '_dots'].push(x); rec[line + '_orig'].push(1); n++;
      });
      rec['n_prop_' + line] = n;
      sortLine(rec, line);
    });
    rec.proposals_seen = true;
    rec.proposal_sig = proposalSig(img);
    rec.proposal_method = p.method || null;
    stamp(rec);
  }

  function removeProposalDots(rec) {
    ['h', 'v'].forEach((line) => {
      const d = rec[line + '_dots'], o = rec[line + '_orig'];
      const keep = d.map((_, i) => i).filter((i) => !o[i]);
      rec[line + '_dots'] = keep.map((i) => d[i]);
      rec[line + '_orig'] = keep.map((i) => o[i]);
    });
    touch(rec);
  }

  /**
   * Blind ON removes proposal-origin dots (user dots stay); OFF re-inserts proposals
   * (skipping near-duplicates of the user's dots). Every toggle is logged in blind_log and
   * proposals_seen stays true once proposals were shown, so blind counts made after seeing
   * the proposals can be excluded from the anchoring analysis.
   */
  function setBlind(rec, img, on) {
    on = !!on;
    if (on === rec.blind_mode) return;
    rec.blind_mode = on;
    if (!Array.isArray(rec.blind_log)) rec.blind_log = [];
    rec.blind_log.push({ t: nowIso(), on: on });
    if (on) removeProposalDots(rec);
    else insertProposals(rec, img, null);
    touch(rec);
  }

  /** Untouched since creation (no user edit of any kind, not done)? Legacy records: no. */
  function isPristine(rec) {
    return rec.n_edits === 0 && !rec.done;
  }

  /**
   * Bring a record up to date with the image's current proposals.
   * Returns 'same' (nothing to do), 'refreshed' (record was untouched, re-initialised
   * in place from the new proposals) or 'stale' (newer proposals exist but the record has
   * user edits, which are never overwritten; "Reset to proposals" switches explicitly).
   */
  function refreshProposals(rec, img) {
    if (rec.proposal_sig === proposalSig(img)) return 'same';
    if (!isPristine(rec)) return 'stale';
    const fresh = newRecord(img, { annotator: rec.annotator, blind: rec.blind_mode, iaa: rec.iaa });
    ['created', 'time_spent_s', 'blind_initial', 'blind_log'].forEach((k) => {
      if (rec[k] !== undefined) fresh[k] = rec[k];
    });
    Object.keys(rec).forEach((k) => { delete rec[k]; });
    Object.assign(rec, fresh);
    return 'refreshed';
  }

  /**
   * Proposals are positions on AXIS-ALIGNED lines, so a reset also turns both lines back to 0 deg, and
   * (v2.2.1) a turned image back to 0 deg (one 'reset' entry per line with view 0).
   */
  function resetToProposals(rec, img) {
    const p = img.proposal || {};
    const S = rec.rect_size;
    ['h', 'v'].forEach((l) => { rec[l + '_dots'] = []; rec[l + '_orig'] = []; });
    rec.h_line_y = r1(clamp(p.h_line_y != null ? p.h_line_y : S / 2, 0, S));
    rec.v_line_x = r1(clamp(p.v_line_x != null ? p.v_line_x : S / 2, 0, S));
    if (viewAngle(rec) !== 0) {
      rec.view_rot_deg = 0;
      ['h', 'v'].forEach((l) => { rec[l + '_angle_deg'] = 0; logAngle(rec, l, 0, 'reset', { view: 0 }); });
    } else ['h', 'v'].forEach((l) => { if (lineAngle(rec, l) !== 0) setAngle(rec, l, 0, 'reset'); });
    rec.proposal_method = p.method || null;
    rec.proposal_sig = proposalSig(img);
    rec.n_prop_h = 0; rec.n_prop_v = 0;
    if (!rec.blind_mode) insertProposals(rec, img);
    touch(rec);
  }

  function clearLine(rec, line) {
    rec[line + '_dots'] = []; rec[line + '_orig'] = [];
    touch(rec);
  }

  /** Add a user dot at parameter `pos` (clamped to the line's segment inside the window). */
  function addDot(rec, line, pos) {
    const rg = paramRange(rec, line);
    const x = r1(clamp(pos, rg.lo, rg.hi));
    const d = rec[line + '_dots'];
    let i = 0;
    while (i < d.length && d[i] < x) i++;
    d.splice(i, 0, x);
    rec[line + '_orig'].splice(i, 0, 0);
    touch(rec);
    return i;
  }

  function removeDot(rec, line, index) {
    rec[line + '_dots'].splice(index, 1);
    rec[line + '_orig'].splice(index, 1);
    touch(rec);
  }

  /** Nearest dot on `line` to position `pos` -> {index, dist} or null. */
  function nearestDot(rec, line, pos) {
    const d = rec[line + '_dots'];
    let best = null;
    for (let i = 0; i < d.length; i++) {
      const dist = Math.abs(d[i] - pos);
      if (!best || dist < best.dist) best = { index: i, dist: dist };
    }
    return best;
  }

  /** World (x, y) of every dot: it lies on its (possibly rotated) line. */
  function dotXY(rec, line, i) {
    return pointAt(rec, line, rec[line + '_dots'][i]);
  }

  /** Delete all dots inside the world-space box; returns number removed. */
  function boxDelete(rec, x0, y0, x1, y1) {
    const xa = Math.min(x0, x1), xb = Math.max(x0, x1), ya = Math.min(y0, y1), yb = Math.max(y0, y1);
    let n = 0;
    ['h', 'v'].forEach((line) => {
      const d = rec[line + '_dots'], o = rec[line + '_orig'];
      const keep = [];
      for (let i = 0; i < d.length; i++) {
        const [x, y] = dotXY(rec, line, i);
        if (x >= xa && x <= xb && y >= ya && y <= yb) n++;
        else keep.push(i);
      }
      rec[line + '_dots'] = keep.map((i) => d[i]);
      rec[line + '_orig'] = keep.map((i) => o[i]);
    });
    if (n) touch(rec);
    return n;
  }

  /** Delete the dots for which pred(x, y, line, index) is true (x, y = image position); returns the number removed. */
  function removeDotsWhere(rec, pred) {
    let n = 0;
    ['h', 'v'].forEach((line) => {
      const d = rec[line + '_dots'], o = rec[line + '_orig'];
      const keep = [];
      for (let i = 0; i < d.length; i++) {
        const xy = dotXY(rec, line, i);
        if (pred(xy[0], xy[1], line, i)) n++; else keep.push(i);
      }
      rec[line + '_dots'] = keep.map((i) => d[i]);
      rec[line + '_orig'] = keep.map((i) => o[i]);
    });
    if (n) touch(rec);
    return n;
  }
  /**
   * Box delete in SCREEN space (v2.2.1): deletes the dots whose screen position (viewToScreen with
   * `view`) lies in the screen rectangle (x0, y0)-(x1, y1). With a turned image the box is a turned
   * rectangle in the image, so it cannot be done with an image-space box.
   */
  function boxDeleteScreen(rec, view, x0, y0, x1, y1) {
    const xa = Math.min(x0, x1), xb = Math.max(x0, x1), ya = Math.min(y0, y1), yb = Math.max(y0, y1), S = rec.rect_size;
    return removeDotsWhere(rec, (x, y) => {
      const p = viewToScreen(view, S, x, y);
      return p[0] >= xa && p[0] <= xb && p[1] >= ya && p[1] <= yb;
    });
  }

  /** Remove dots that lie outside the window (possible only after rotating a line); returns n. */
  function removeOutside(rec, line) {
    const L = lineInfo(rec, line);
    if (!L.nOut) return 0;
    const d = rec[line + '_dots'], o = rec[line + '_orig'];
    const keep = d.map((_, i) => i).filter((i) => d[i] >= L.lo - 0.5 && d[i] <= L.hi + 0.5);
    rec[line + '_dots'] = keep.map((i) => d[i]);
    rec[line + '_orig'] = keep.map((i) => o[i]);
    touch(rec);
    return L.nOut;
  }

  /** Move the line's pivot (H: y, V: x) — a translation; the angle is kept. */
  function moveLine(rec, line, pos) {
    const v = r1(clamp(pos, 0, rec.rect_size));
    if (line === 'h') rec.h_line_y = v; else rec.v_line_x = v;
    touch(rec);
  }

  /** Raw dot counts (v1 semantics; = threads/inch only for unrotated lines). */
  function counts(rec) {
    return { h: rec.h_dots.length, v: rec.v_dots.length };
  }
  /** Normalised threads per inch of both lines (== counts() for unrotated lines). */
  function tpis(rec) {
    return { h: lineInfo(rec, 'h').tpi, v: lineInfo(rec, 'v').tpi };
  }

  /** Warp ends/inch and weft picks/inch from the raw counts, if the warp direction is known. */
  function warpWeft(rec) {
    if (rec.warp_dir === 'vertical') return { warp: rec.h_dots.length, weft: rec.v_dots.length };
    if (rec.warp_dir === 'horizontal') return { warp: rec.v_dots.length, weft: rec.h_dots.length };
    return { warp: null, weft: null };
  }
  /** Same, from the normalised TPI. */
  function warpWeftTpi(rec) {
    const t = tpis(rec);
    if (rec.warp_dir === 'vertical') return { warp: t.h, weft: t.v };
    if (rec.warp_dir === 'horizontal') return { warp: t.v, weft: t.h };
    return { warp: null, weft: null };
  }

  function median(a) {
    if (!a.length) return NaN;
    const s = a.slice().sort((x, y) => x - y);
    const m = s.length >> 1;
    return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
  }

  /**
   * Spacing sanity check for one line. Returns null for < 3 dots, else
   * {median, est, big:[[a,b],...], small:[[a,b],...]} where `big` gaps (> 1.6x median,
   * incl. segment ends lo/hi > 1.5x median) suggest a missed yarn and `small` gaps
   * (< 0.45x median) suggest a duplicate; est = S / (median spacing along the line).
   * lo/hi default to 0/S and k (line length per unit parameter, 1/cos a) to 1, i.e. v1.
   */
  function gapStats(dots, S, lo, hi, k) {
    if (!dots || dots.length < 3) return null;
    lo = lo == null ? 0 : lo; hi = hi == null ? S : hi; k = k || 1;
    const g = [];
    for (let i = 1; i < dots.length; i++) g.push(dots[i] - dots[i - 1]);
    const m = median(g);
    if (!(m > 0)) return null;
    const big = [], small = [];
    for (let i = 1; i < dots.length; i++) {
      const gap = dots[i] - dots[i - 1];
      if (gap > 1.6 * m) big.push([dots[i - 1], dots[i]]);
      else if (gap < 0.45 * m) small.push([dots[i - 1], dots[i]]);
    }
    if (dots[0] - lo > 1.5 * m) big.unshift([lo, dots[0]]);
    if (hi - dots[dots.length - 1] > 1.5 * m) big.push([dots[dots.length - 1], hi]);
    return { median: m, est: S / (m * k), big: big, small: small };
  }
  /** gapStats for a record's line, using the line's own segment and angle. */
  function lineGapStats(rec, line) {
    const L = lineInfo(rec, line);
    return gapStats(rec[line + '_dots'], rec.rect_size, L.lo, L.hi, L.k);
  }

  /** Every user edit goes through touch(): new timestamp (newest wins on merge) + edit count. */
  function touch(rec) { rec.n_edits = (rec.n_edits || 0) + 1; rec.timestamp = nowIso(); updateDerived(rec); }
  /** Non-user change (e.g. proposals inserted on creation): timestamp only. */
  function stamp(rec) { rec.timestamp = nowIso(); updateDerived(rec); }

  // ------------------------------------------------------------------ orientation estimate (v2)
  function fftPlan(n) {
    let levels = 0;
    while ((1 << levels) < n) levels++;
    if ((1 << levels) !== n) throw new Error('FFT size must be a power of 2');
    const cosT = new Float64Array(n / 2), sinT = new Float64Array(n / 2), rev = new Uint32Array(n);
    for (let i = 0; i < n / 2; i++) { cosT[i] = Math.cos(2 * Math.PI * i / n); sinT[i] = Math.sin(2 * Math.PI * i / n); }
    for (let i = 0; i < n; i++) {
      let x = i, y = 0;
      for (let j = 0; j < levels; j++) { y = (y << 1) | (x & 1); x >>>= 1; }
      rev[i] = y;
    }
    return { n: n, cosT: cosT, sinT: sinT, rev: rev };
  }
  /** In-place radix-2 forward DFT (sign -1) of (re, im), both of length plan.n. */
  function fft(plan, re, im) {
    const n = plan.n, rev = plan.rev, cosT = plan.cosT, sinT = plan.sinT;
    for (let i = 0; i < n; i++) {
      const j = rev[i];
      if (j > i) { let t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; }
    }
    for (let size = 2; size <= n; size <<= 1) {
      const half = size >> 1, step = n / size;
      for (let i = 0; i < n; i += size) {
        for (let j = i, k = 0; j < i + half; j++, k += step) {
          const l = j + half, c = cosT[k], s = sinT[k];
          const tr = re[l] * c + im[l] * s, ti = im[l] * c - re[l] * s;
          re[l] = re[j] - tr; im[l] = im[j] - ti;
          re[j] += tr; im[j] += ti;
        }
      }
    }
  }

  /**
   * Peak of an angle histogram h (bins of `bin` deg from -W). `at` (optional) = index of
   * the chosen peak; it is refined by hill-climbing to the local maximum and a parabola.
   * ratio = peak height / highest other local maximum >= 2 deg away (< 1: not the
   * strongest direction), share = fraction of the votes within +-1 deg.
   */
  function histPeak(h, bin, W, at) {
    const n = h.length;
    let best = -1;
    if (at == null) { for (let i = 0; i < n; i++) if (best < 0 || h[i] > h[best]) best = i; }
    else {   // best in +-2 deg, then climb to the local maximum (at most 3 deg further)
      const R = Math.round(2 / bin), M = Math.round(3 / bin);
      for (let i = Math.max(0, at - R); i <= Math.min(n - 1, at + R); i++) if (best < 0 || h[i] > h[best]) best = i;
      for (let k = 0; k < M; k++) {
        if (best > 0 && h[best - 1] > h[best]) best--;
        else if (best < n - 1 && h[best + 1] > h[best]) best++;
        else break;
      }
    }
    if (best < 0 || !(h[best] > 0)) return null;
    let off = 0;
    if (best > 0 && best < n - 1) {
      const a = h[best - 1], b = h[best], c = h[best + 1], den = a - 2 * b + c;
      if (den < 0) off = clamp(0.5 * (a - c) / den, -0.5, 0.5);
    }
    const sep = Math.round(2 / bin);
    let second = 0;
    for (let i = 1; i < n - 1; i++) {
      if (Math.abs(i - best) < sep) continue;
      if (h[i] >= h[i - 1] && h[i] >= h[i + 1] && h[i] > second) second = h[i];
    }
    let tot = 0, near = 0;
    for (let i = 0; i < n; i++) { tot += h[i]; if (Math.abs(i - best) <= Math.round(1 / bin)) near += h[i]; }
    const deg = -W + (best + off) * bin;
    const ratio = second > 0 ? h[best] / second : Infinity;
    const share = tot > 0 ? near / tot : 0;
    return { deg: r2(deg), ratio: ratio, share: share, edge: best === 0 || best === n - 1 };
  }

  /**
   * Dominant yarn orientation of both yarn families from a grey image.
   * gray: Float32Array/Float64Array/Array of N*N values (row-major, N a power of 2) that
   * covers the 1-inch window, so frequency radius r = cycles per inch.
   * Method: Hann-windowed 2-D power spectrum, whitened per frequency ring (geometric
   * mean), then an angle histogram (0.1 deg bins; each bin with whitened power z > 6 votes
   * sqrt(z) - sqrt(6)) of all bins with rMin <= r <= rMax, within +-18 deg of each axis.
   * The family crossed by the H line (vertical yarns) has its spectral peaks along a ray
   * near the u axis at angle = the H line angle that is perpendicular to those yarns;
   * likewise near the v axis for the V line. Harmonics lie on the same ray, so they all
   * vote for the same angle. A mild prior (votes x (1 - 0.5 (a/18)^2)) and a joint choice
   * of the two families' peaks (penalty 0.08 per degree of |h - v| beyond 4, as a rigidly
   * rotated fabric has equal angles) suppress twill-diagonal harmonics. On the 150 crops it
   * agrees with the lattice method's angles within 2 deg for 133/147 H and 130/147 V lines.
   * -> {h: {deg, ratio, share, clear, edge}, v: {...}}  deg in the tool's line-angle
   *    convention; null for a family without signal.
   */
  function estimateOrientation(gray, N, opts) {
    opts = opts || {};
    const W = opts.window || 18, bin = opts.bin || 0.1;
    const rMin = opts.rMin || 6, rMax = Math.min(opts.rMax || Infinity, N / 2 - 1);
    const T = opts.thresh || 6;
    const wmode = opts.weight || 'sqrt';                  // vote per bin: sqrt(z) - sqrt(T) ('log', 'lin' also)
    const prior = opts.prior == null ? 0.5 : opts.prior;  // tilt prior: votes *= 1 - prior (a/W)^2
    const skew = opts.skew == null ? 0.08 : opts.skew;    // joint H/V choice: penalty per degree of |h - v| beyond 4
    const re = new Float64Array(N * N), im = new Float64Array(N * N);
    let mean = 0;
    for (let i = 0; i < N * N; i++) mean += gray[i];
    mean /= N * N;
    const hann = new Float64Array(N);
    for (let i = 0; i < N; i++) hann[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * (i + 0.5) / N);
    for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) re[y * N + x] = (gray[y * N + x] - mean) * hann[x] * hann[y];
    const plan = fftPlan(N);
    for (let y = 0; y < N; y++) fft(plan, re.subarray(y * N, y * N + N), im.subarray(y * N, y * N + N));
    const cr = new Float64Array(N), ci = new Float64Array(N);
    for (let x = 0; x < N; x++) {
      for (let y = 0; y < N; y++) { cr[y] = re[y * N + x]; ci[y] = im[y * N + x]; }
      fft(plan, cr, ci);
      for (let y = 0; y < N; y++) { re[y * N + x] = cr[y]; im[y * N + x] = ci[y]; }
    }
    // power (into re) + per-ring geometric-mean background
    const nRing = Math.ceil(N * 0.75) + 2;
    const ringSum = new Float64Array(nRing), ringN = new Float64Array(nRing);
    let pmax = 0;
    for (let i = 0; i < N * N; i++) { const p = re[i] * re[i] + im[i] * im[i]; re[i] = p; if (p > pmax) pmax = p; }
    const eps = pmax * 1e-14 + 1e-30;
    const fu = (c) => (c < N / 2 ? c : c - N);
    for (let y = 0; y < N; y++) {
      const v = fu(y);
      for (let x = 0; x < N; x++) {
        const u = fu(x), ri = Math.round(Math.sqrt(u * u + v * v));
        if (ri < nRing) { ringSum[ri] += Math.log(re[y * N + x] + eps); ringN[ri]++; }
      }
    }
    const ringBg = new Float64Array(nRing);
    for (let i = 0; i < nRing; i++) ringBg[i] = Math.exp(ringSum[i] / Math.max(1, ringN[i]));
    const nb = Math.round(2 * W / bin) + 1;
    const hh = new Float64Array(nb), hv = new Float64Array(nb);     // selection votes (wmode)
    const lh = new Float64Array(nb), lv = new Float64Array(nb);     // measurement votes (log)
    const add = (h, a, w) => {
      const f = (a + W) / bin, i0 = Math.floor(f), t = f - i0;
      if (i0 >= 0 && i0 < nb) h[i0] += w * (1 - t);
      if (i0 + 1 >= 0 && i0 + 1 < nb) h[i0 + 1] += w * t;
    };
    for (let y = 0; y < N; y++) {
      const v = fu(y);
      for (let x = 0; x < N; x++) {
        const u = fu(x);
        if (!(v > 0 || (v === 0 && u > 0))) continue;          // half plane (P is symmetric)
        const r = Math.sqrt(u * u + v * v);
        if (r < rMin || r > rMax) continue;
        const z = (re[y * N + x] + eps) / ringBg[Math.round(r)];
        if (z <= T) continue;
        const wl = Math.log(z / T);
        const w = wmode === 'lin' ? z - T : wmode === 'sqrt' ? Math.sqrt(z) - Math.sqrt(T) : wl;
        let a = Math.atan2(v, u) / D2R;                        // (-180, 180], y down
        if (a > 90) a -= 180; else if (a <= -90) a += 180;     // fold to (-90, 90]
        if (Math.abs(a) <= W) { add(hh, a, w); add(lh, a, wl); }
        let b = a - 90;                                        // V line angle
        if (b <= -90) b += 180;
        if (Math.abs(b) <= W) { add(hv, b, w); add(lv, b, wl); }
      }
    }
    // light smoothing (sigma 0.25 deg)
    const smooth = (h) => {
      const s = (opts.smooth || 0.25) / bin, R = Math.ceil(3 * s), k = [];
      for (let i = -R; i <= R; i++) k.push(Math.exp(-0.5 * (i / s) * (i / s)));
      const out = new Float64Array(h.length);
      for (let i = 0; i < h.length; i++) {
        let acc = 0, wsum = 0;
        for (let j = -R; j <= R; j++) { const q = i + j; if (q >= 0 && q < h.length) { acc += h[q] * k[j + R]; wsum += k[j + R]; } }
        out[i] = acc / wsum;
      }
      return out;
    };
    // choose the peak on prior-weighted histograms, then measure it on the raw ones (no bias)
    // (votes weighted by sqrt(z) pick the right peak more often on twills; log(z) votes spread
    // over the whole main lobe and measure its angle without bin quantisation)
    const raw = {}, wtd = {}, pick = {}, meas = { h: smooth(lh), v: smooth(lv) };
    [['h', hh], ['v', hv]].forEach(([line, h]) => {
      const s = smooth(h), w = s.slice();
      if (prior) for (let i = 0; i < w.length; i++) { const a = -W + i * bin; w[i] *= Math.max(0, 1 - prior * (a / W) * (a / W)); }
      raw[line] = s; wtd[line] = w;
      let b = 0;
      for (let i = 1; i < w.length; i++) if (w[i] > w[b]) b = i;
      pick[line] = w[b] > 0 ? b : null;
    });
    if (skew && pick.h != null && pick.v != null) {
      // a rigidly rotated fabric has equal H and V angles (perspective adds a few degrees of
      // skew): choose the pair of strong local maxima with the best normalised height minus
      // a penalty on |h - v| beyond 4 degrees
      const peaks = (w) => {
        let mx = 0; for (let i = 0; i < w.length; i++) if (w[i] > mx) mx = w[i];
        const out = [];
        for (let i = 1; i < w.length - 1; i++) if (w[i] >= w[i - 1] && w[i] > w[i + 1] && w[i] >= 0.2 * mx) out.push({ i: i, a: -W + i * bin, n: w[i] / mx });
        return out.sort((x, y) => y.n - x.n).slice(0, 6);
      };
      let best = null;
      peaks(wtd.h).forEach((a) => peaks(wtd.v).forEach((b) => {
        const sc = a.n + b.n - skew * Math.max(0, Math.abs(a.a - b.a) - 4);
        if (!best || sc > best.sc) best = { sc: sc, h: a.i, v: b.i };
      }));
      if (best) { pick.h = best.h; pick.v = best.v; }
    }
    const res = {};
    ['h', 'v'].forEach((line) => {
      if (pick[line] == null) { res[line] = null; return; }
      const pk = histPeak(raw[line], bin, W, pick[line]);            // ratio / share / edge
      const m = histPeak(meas[line], bin, W, pick[line]);            // angle
      if (pk && m) { pk.deg = m.deg; pk.edge = pk.edge || m.edge; }
      if (pk) pk.clear = pk.ratio >= 1.5 && !pk.edge;
      res[line] = pk;
    });
    // after rectification the two families are rarely more than a few degrees apart: a larger
    // skew usually means one family locked onto a twill harmonic -> ask the annotator to check
    if (res.h && res.v && Math.abs(res.h.deg - res.v.deg) > (opts.maxSkew || 8)) {
      res.h.clear = false; res.v.clear = false; res.h.skewed = true; res.v.skewed = true;
    }
    res.N = N; res.rMin = rMin; res.rMax = rMax; res.window = W;
    return res;
  }

  // ------------------------------------------------------------------ scenes (v2.1)
  // The 150 files are 76 scenes (prepare.py / scenes.py). Each image of a v2.1 manifest has
  // scene_id (= id of the scene's representative), is_scene_rep, scene_members, kind and
  // scene_rot_k (crop(file) ~= rot90(crop(rep), k), counter-clockwise). Only reps are annotated;
  // the other files are read-only copies whose labels are copied from the rep (h <-> v swapped
  // for odd k). Nothing here changes the record format.
  const KIND_LABEL = {
    rep: 'scene representative', byte_identical: 'byte-identical copy', quarter_res_copy: 'quarter-resolution copy',
    png_copy: 'PNG copy', notebook_synthetic: 'synthetic copy (2024 notebook)',
  };
  const WARP_SWAP = { vertical: 'horizontal', horizontal: 'vertical' };

  /** True when every manifest image carries the v2.1 scene fields. */
  function hasScenes(manifest) {
    const im = manifest && manifest.images;
    return !!(im && im.length && im.every((e) => e && e.scene_id != null && typeof e.is_scene_rep === 'boolean'));
  }
  /** A non-representative file of a scene (labels copied, read-only by default). */
  function isCopy(img) { return !!(img && img.scene_id != null && img.is_scene_rep === false); }
  /** Id of the image's scene representative (itself for a rep or a manifest without scenes). */
  function sceneRepId(img) { return img ? (img.scene_id != null ? img.scene_id : img.id) : null; }
  /** Rotation of the file's crop relative to its rep's crop, in quarter turns (0..3). */
  function sceneRotK(img) { const k = Number(img && img.scene_rot_k); return isFinite(k) ? ((k % 4) + 4) % 4 : 0; }
  /**
   * The rep's labels as they apply to copy `img`: threads/inch per line and warp direction,
   * h <-> v swapped when the copy's crop is the rep's turned by an odd number of quarter turns
   * (same rule as propagate_scene_labels.py). -> null without a rep record.
   */
  function copiedLabel(repRec, img) {
    if (!repRec || !Array.isArray(repRec.h_dots) || !Array.isArray(repRec.v_dots) || !repRec.rect_size) return null;
    const t = tpis(repRec), k = sceneRotK(img), swap = k % 2 === 1, wd = repRec.warp_dir;
    return {
      from: repRec.id, k: k, swap: swap, done: !!repRec.done,
      h: swap ? t.v : t.h, v: swap ? t.h : t.v,
      h_count: swap ? repRec.v_dots.length : repRec.h_dots.length,
      v_count: swap ? repRec.h_dots.length : repRec.v_dots.length,
      weave: repRec.weave || null, warp_dir: swap && WARP_SWAP[wd] ? WARP_SWAP[wd] : (wd || null),
    };
  }

  /** Quality flags / low confidence / unsure weave / non-woven / notes. */
  function isFlagged(rec) {
    if (!rec) return false;
    const f = rec.flags || {};
    return Object.keys(f).some((k) => f[k]) || rec.confidence === 1 || rec.weave === 'unsure' ||
      !!rec.nonwoven || !!(rec.notes && String(rec.notes).trim().length > 0);
  }
  /**
   * Ids shown by a "Show" filter. With scenes: 'scenes' = the reps (the photos to annotate) and
   * 'todo' = reps not done (copies are never "to do"); 'all', 'done', 'iaa' and 'flagged' are
   * unchanged. Without scenes 'scenes' falls back to 'all' (v2 behaviour).
   * v2.2: 'attention' = attentionOrder() (NOT in id order; opts = {policy, manualDefault}).
   */
  function filterIds(images, records, filter, iaaSet, scenes, opts) {
    const rec = (id) => records[String(id)];
    if (filter === 'attention') return attentionOrder(images, records, iaaSet, opts);
    return images.filter((im) => {
      const r = rec(im.id);
      if (filter === 'scenes') return !scenes || !isCopy(im);
      if (filter === 'todo') return !(r && r.done) && !(scenes && isCopy(im));
      if (filter === 'done') return !!(r && r.done);
      if (filter === 'iaa') return iaaSet.has(im.id);
      if (filter === 'flagged') return isFlagged(r);
      return true;
    }).map((im) => im.id);
  }
  /** Progress: with scenes counted over the reps (n / 76 scenes), else over all images. */
  function progress(images, records, iaaSet, scenes) {
    const units = scenes ? images.filter((im) => !isCopy(im)) : images;
    const isDone = (im) => { const r = records[String(im.id)]; return !!(r && r.done); };
    const iaa = images.filter((im) => iaaSet.has(im.id));
    return { unit: scenes ? 'scenes' : 'images', done: units.filter(isDone).length, total: units.length,
      iaaDone: iaa.filter(isDone).length, iaaTotal: iaaSet.size };
  }
  /**
   * The IAA subset can change (prepare.py v2 re-draw). A record that was opened but never edited
   * (pristine) is re-created for its current membership and blind policy, like refreshProposals;
   * the audit trail stays honest: created, active time, blind_initial and blind_log are kept and
   * proposals_seen stays true once proposals were shown. Edited records are never changed.
   * Returns true if the record was re-created.
   */
  function syncIaa(rec, img, inIaa, opts) {
    opts = opts || {};
    if (!rec || !!rec.iaa === !!inIaa || !isPristine(rec)) return false;
    const blind = blindForPolicy(opts.policy || 'iaa', img.id, opts.manualDefault, inIaa);
    const fresh = newRecord(img, { annotator: rec.annotator, blind: blind, iaa: inIaa });
    fresh.created = rec.created || fresh.created;
    fresh.time_spent_s = rec.time_spent_s || 0;
    if (rec.blind_initial !== undefined) fresh.blind_initial = rec.blind_initial;
    fresh.blind_log = Array.isArray(rec.blind_log) ? rec.blind_log.slice() : [];
    if (!!rec.blind_mode !== !!fresh.blind_mode) fresh.blind_log.push({ t: nowIso(), on: !!fresh.blind_mode, iaa_resync: true });
    fresh.proposals_seen = !!(rec.proposals_seen || fresh.proposals_seen);
    Object.keys(rec).forEach((k) => { delete rec[k]; });
    Object.assign(rec, fresh);
    return true;
  }

  // ------------------------------------------------------------------ attention (v2.2)
  // prepare.py --attention-only (attention.py) gives every scene rep `attention` (low | medium | high)
  // and generic `attention_reasons` derived from existing predictions only: estimator disagreement,
  // the counter's confidence, resolvability, glare and px per yarn. No count is in the manifest fields
  // and none is ever shown. Nothing is shown on IAA-subset scenes or on blind records, so a blind count
  // never sees a machine signal; copies show nothing (they are not annotated).
  const ATTN_RANK = { low: 0, medium: 1, high: 2 };
  const ATTN_PREFIX = { high: 'Check carefully', medium: 'Take care' };

  /** True when at least one manifest image carries an attention level. */
  function hasAttention(manifest) {
    const im = manifest && manifest.images;
    return !!(im && im.some((e) => e && Object.prototype.hasOwnProperty.call(ATTN_RANK, e.attention)));
  }
  /** {level, rank, reasons} of an image, or null (no or invalid level). Reasons are strings only. */
  function attentionOf(img) {
    if (!img || !Object.prototype.hasOwnProperty.call(ATTN_RANK, img.attention)) return null;
    const reasons = Array.isArray(img.attention_reasons)
      ? img.attention_reasons.filter((x) => typeof x === 'string' && x.trim()).map((x) => x.trim()) : [];
    return { level: img.attention, rank: ATTN_RANK[img.attention], reasons: reasons };
  }
  /**
   * May the attention of `img` be shown? Never on a copy, never on an IAA-subset scene (also when an
   * IAA scene's copy is unlocked) and never on a blind record. Without a record (an image not opened
   * yet) the blind setting it will open with is predicted from the policy (opts.policy, default 'iaa';
   * opts.manualDefault for 'manual'), as app.js ensureRec() does.
   */
  function attentionVisible(img, rec, iaaSet, opts) {
    if (!attentionOf(img) || isCopy(img)) return false;
    const inIaa = !!(iaaSet && (iaaSet.has(img.id) || iaaSet.has(sceneRepId(img))));
    if (inIaa) return false;
    if (rec) return !rec.blind_mode;
    opts = opts || {};
    return !blindForPolicy(opts.policy || 'iaa', img.id, opts.manualDefault, inIaa);
  }
  /**
   * What the tool shows: null for low (nothing), else {level, badge, line, reasons} with
   * badge = "Check carefully: <first reason>" (high) or "Take care: <first reason>" (medium) and
   * line = all reasons. Callers must check attentionVisible() first.
   */
  function attentionDisplay(img) {
    const a = attentionOf(img);
    if (!a || a.rank === 0) return null;
    const first = a.reasons[0] || 'automatic checks flag this photo';
    const more = a.reasons.length > 1 ? ' (+' + (a.reasons.length - 1) + ' more)' : '';
    return { level: a.level, reasons: a.reasons, badge: ATTN_PREFIX[a.level] + ': ' + first + more,
      line: a.reasons.length ? a.reasons.join(' · ') : first };
  }
  /**
   * "High attention first": the reps whose attention may be shown, high -> medium -> low; within a
   * level the higher attention_score first, then by id.
   */
  function attentionOrder(images, records, iaaSet, opts) {
    const rec = (id) => records[String(id)];
    const sc = (im) => (isFinite(Number(im.attention_score)) ? Number(im.attention_score) : 0);
    return images.filter((im) => attentionVisible(im, rec(im.id), iaaSet, opts))
      .sort((a, b) => (ATTN_RANK[b.attention] - ATTN_RANK[a.attention]) || (sc(b) - sc(a)) || (a.id - b.id))
      .map((im) => im.id);
  }

  // ------------------------------------------------------------------ examples panel (v2.2.2)
  // The Examples panel (key X) has two tabs. "How it looks": figures of guide/examples/examples.json
  // (make_tool_examples.py: machine proposals on non-IAA scene reps). "My finished scenes": the current
  // annotator's own done records. Blind protocol: a figure never shows an IAA-subset scene (nor a copy of one)
  // and is hidden while one of its scenes is, or will open, blind for this annotator; the gallery shows only
  // records whose annotator field is the current Annotator name (never another annotator's), only done ones,
  // and for IAA-subset scenes neither dots nor counts. Read-only: no record is changed.
  const EXAMPLE_SRC = /^guide\/(?:[A-Za-z0-9_-]+\/)*[A-Za-z0-9_.-]+\.(?:png|jpe?g|webp)$/;

  function idMap(images) {
    const m = {};
    (images || []).forEach((im) => { if (im && im.id != null) m[Number(im.id)] = im; });
    return m;
  }
  /** True when the file or its scene is in the IAA subset. */
  function inIaaScene(img, iaaSet) {
    return !!(img && iaaSet && (iaaSet.has(img.id) || iaaSet.has(sceneRepId(img))));
  }
  /** Same annotator: both names non-empty and equal as stored (safeName, the browser / disk key). */
  function sameAnnotator(a, b) {
    const x = String(a == null ? '' : a).trim(), y = String(b == null ? '' : b).trim();
    return !!x && !!y && safeName(x) === safeName(y);
  }
  /** Is (or will) `img` be counted blind by this annotator: its record's blind mode, else the blind policy (as ensureRec). */
  function opensBlind(img, records, iaaSet, opts) {
    const r = records && records[String(img.id)];
    if (r) return !!r.blind_mode;
    opts = opts || {};
    return blindForPolicy(opts.policy || 'iaa', img.id, opts.manualDefault, inIaaScene(img, iaaSet));
  }
  /**
   * Figures of an examples index that may be shown -> {shown: [figure], blind: n hidden (a scene counted blind),
   * refused: n never shown}. Refused: no `scenes` list, an unknown id, a copy, an IAA-subset scene (or a copy of
   * one), or a src outside guide/ (only plain image paths). opts = {policy, manualDefault} (blind policy).
   */
  function visibleExamples(index, images, records, iaaSet, opts) {
    const figs = index && Array.isArray(index.figures) ? index.figures : [];
    const byId = idMap(images), shown = [];
    let blind = 0, refused = 0;
    figs.forEach((f) => {
      const sc = f && Array.isArray(f.scenes) ? f.scenes.map(Number) : [];
      const ok = f && typeof f.src === 'string' && EXAMPLE_SRC.test(f.src) && f.src.indexOf('..') < 0 && sc.length > 0 &&
        sc.every((id) => { const im = byId[id]; return !!im && !isCopy(im) && !inIaaScene(im, iaaSet); });
      if (!ok) { refused++; return; }
      if (sc.some((id) => opensBlind(byId[id], records, iaaSet, opts))) { blind++; return; }
      shown.push(f);
    });
    return { shown: shown, blind: blind, refused: refused };
  }
  /**
   * "My finished scenes": the done records of `annotator` (records whose annotator field is that name; a record
   * of another or of no annotator is never listed), most recent first (timestamp, then id).
   * -> {needName (no annotator name: show a prompt, no gallery), items, nDone (scene reps among them), total (reps)}.
   * item = {id, rep, copy, iaa, t, timestamp, weave, confidence, flags: [keys], nonwoven, empty: lines without dots,
   * tpi: {h, v}, angle: {h, v}, view}; tpi / angle / view are left out for IAA-subset scenes (also when the record's
   * own iaa flag is set), whose dots are never drawn in the gallery.
   */
  function finishedScenes(records, annotator, images, iaaSet) {
    const byId = idMap(images);
    const total = (images || []).filter((im) => !isCopy(im)).length;
    const name = String(annotator == null ? '' : annotator).trim();
    if (!name) return { needName: true, items: [], nDone: 0, total: total };
    const items = [];
    Object.keys(records || {}).forEach((k) => {
      const r = records[k];
      if (!isRecord(r) || !r.done || !sameAnnotator(r.annotator, name)) return;
      const img = byId[Number(r.id)];
      if (!img) return;
      const fl = r.flags || {};
      const it = {
        id: img.id, rep: sceneRepId(img), copy: isCopy(img), iaa: inIaaScene(img, iaaSet) || r.iaa === true,
        t: Date.parse(r.timestamp) || 0, timestamp: r.timestamp || null,
        weave: r.weave || null, confidence: r.confidence == null ? null : Number(r.confidence),
        flags: FLAGS.filter((f) => fl[f.key]).map((f) => f.key), nonwoven: !!r.nonwoven,
        empty: ['h', 'v'].filter((l) => !r[l + '_dots'].length),
      };
      if (!it.iaa && r.rect_size) {
        const h = lineInfo(r, 'h'), v = lineInfo(r, 'v');
        it.tpi = { h: h.tpi, v: v.tpi }; it.angle = { h: h.angle, v: v.angle }; it.view = viewAngle(r);
      }
      items.push(it);
    });
    items.sort((a, b) => (b.t - a.t) || (a.id - b.id));
    return { needName: false, items: items, nDone: items.filter((it) => !it.copy).length, total: total };
  }
  /**
   * Geometry of a gallery thumbnail, `size` px square: the crop turned by the record's image turn and fitted with
   * `pad` px margin (as fitView), the window outline, both line segments at their angles and every dot, all in
   * thumbnail px through viewToScreen (the main view's mapping). -> {view, size, window, lines: {h, v}, dots: {h, v}}.
   */
  function thumbGeometry(rec, size, pad) {
    const S = rec.rect_size, rot = viewAngle(rec);
    pad = pad || 0;
    const scale = Math.max(1e-6, (size - 2 * pad) / (S * viewExtent(rot)));
    const view = { scale: scale, ox: size / 2 - S / 2 * scale, oy: size / 2 - S / 2 * scale, rot: rot };
    const P = (w) => viewToScreen(view, S, w[0], w[1]);
    const out = { view: view, size: size, window: [[0, 0], [S, 0], [S, S], [0, S]].map(P), lines: {}, dots: {} };
    ['h', 'v'].forEach((l) => {
      out.lines[l] = segment(rec, l).map(P);
      out.dots[l] = (rec[l + '_dots'] || []).map((_, i) => P(dotXY(rec, l, i)));
    });
    return out;
  }

  // ------------------------------------------------------------------ store / IO
  function emptyStore(annotator) {
    return { schema: SCHEMA, annotator: annotator || '', records: {}, updated: nowIso() };
  }

  function isRecord(r) {
    return r && typeof r === 'object' && r.id != null && Array.isArray(r.h_dots) && Array.isArray(r.v_dots);
  }

  /** Normalise a parsed export / store object into {annotator, records:{id:rec}}. */
  function parseImport(obj) {
    if (!obj || typeof obj !== 'object') throw new Error('not a JSON object');
    let recs = obj.records;
    if (Array.isArray(recs)) {
      const m = {};
      recs.forEach((r) => { if (isRecord(r)) m[String(r.id)] = r; });
      recs = m;
    }
    if (!recs || typeof recs !== 'object') throw new Error('no "records" field');
    const out = {};
    Object.keys(recs).forEach((k) => {
      const r = recs[k];
      if (!isRecord(r)) return;
      if (!Array.isArray(r.h_orig) || r.h_orig.length !== r.h_dots.length) r.h_orig = r.h_dots.map(() => 0);
      if (!Array.isArray(r.v_orig) || r.v_orig.length !== r.v_dots.length) r.v_orig = r.v_dots.map(() => 0);
      if (!r.flags) r.flags = emptyFlags();
      if (!Array.isArray(r.blind_log)) r.blind_log = [];
      out[String(r.id)] = r;
    });
    return { annotator: obj.annotator || '', records: out };
  }

  /**
   * Merge `incoming` records into `base` (both {id:rec}). An untouched record (opened but
   * never edited) never replaces an edited one, whatever the timestamps (e.g. a fresh
   * browser re-opening an image must not beat the finished annotation on disk);
   * otherwise the newer timestamp wins and ties keep `base`.
   */
  function mergeRecords(base, incoming) {
    const merged = Object.assign({}, base);
    let added = 0, updated = 0, kept = 0;
    Object.keys(incoming).forEach((k) => {
      const a = base[k], b = incoming[k];
      if (!a) { merged[k] = b; added++; return; }
      const pa = isPristine(a), pb = isPristine(b);
      let take;
      if (pa !== pb) take = pa;  // edited beats untouched
      else take = (Date.parse(b.timestamp) || 0) > (Date.parse(a.timestamp) || 0);
      if (take) { merged[k] = b; updated++; } else kept++;
    });
    return { merged: merged, added: added, updated: updated, kept: kept };
  }

  /** Export (download / disk autosave). Every record carries the v2 derived line fields. */
  function exportJSON(store, manifest) {
    const ids = Object.keys(store.records).map(Number).sort((a, b) => a - b);
    return {
      schema: SCHEMA,
      tool: 'wvaq2027-thread-count-annotator',
      tool_version: TOOL_VERSION,
      exported: nowIso(),
      annotator: store.annotator,
      manifest_generated: manifest ? manifest.generated : null,
      proposals_updated: manifest ? manifest.proposals_updated : null,
      n_records: ids.length,
      n_done: ids.filter((i) => store.records[i].done).length,
      units: 'dot positions are rectified-crop pixels; crop side rect_size == 1 inch; h_count/v_count are dot counts; ' +
        '<line>_tpi = dots on the line segment inside the window / segment length in inches (== count when <line>_angle_deg is 0)',
      records: ids.map((i) => withDerived(store.records[i])),
    };
  }

  // v1 columns first, unchanged; v2 columns APPENDED so old parsers keep working
  const CSV_COLS_V1 = [
    'id', 'annotator', 'done', 'h_count', 'v_count', 'n_dots_h', 'n_dots_v',
    'warp_per_inch', 'weft_per_inch', 'iaa', 'blind_mode', 'blind_initial', 'n_blind_toggles',
    'proposals_seen', 'proposal_method', 'proposal_sig',
    'n_prop_h', 'n_prop_v', 'n_prop_kept_h', 'n_prop_kept_v', 'n_user_h', 'n_user_v',
    'weave', 'warp_dir', 'flags',
  ].concat(FLAGS.map((f) => 'flag_' + f.key)).concat([
    'nonwoven', 'confidence', 'notes', 'rect_size', 'h_line_y', 'v_line_x',
    'spacing_est_h', 'spacing_est_v', 'time_spent_s', 'n_edits', 'created', 'timestamp',
    'h_dots', 'v_dots', 'h_orig', 'v_orig',
  ]);
  const CSV_COLS_V2 = [
    'h_angle_deg', 'v_angle_deg', 'h_seg_len_px', 'v_seg_len_px', 'h_seg_len_in', 'v_seg_len_in',
    'h_raw_count', 'v_raw_count', 'h_tpi', 'v_tpi', 'warp_tpi', 'weft_tpi',
    'h_angle_src', 'v_angle_src', 'tool_version',
  ];
  // v2.2.1: appended at the END only (all earlier columns keep their position)
  const CSV_COLS_V221 = ['view_rot_deg'];
  const CSV_COLS = CSV_COLS_V1.concat(CSV_COLS_V2, CSV_COLS_V221);

  function csvCell(v) {
    if (v === null || v === undefined) return '';
    if (typeof v === 'boolean') return v ? '1' : '0';
    const s = String(v);
    return /[",\n\r]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
  }

  function csvRow(r) {
    const c = counts(r), ww = warpWeft(r);
    const sum = (a) => a.reduce((x, y) => x + (y ? 1 : 0), 0);
    const gh = lineGapStats(r, 'h'), gv = lineGapStats(r, 'v');
    const flags = r.flags || {};
    const dv = derived(r), wt = warpWeftTpi(r);
    const row = {
      id: r.id, annotator: r.annotator, done: !!r.done,
      h_count: c.h, v_count: c.v, n_dots_h: r.h_dots.length, n_dots_v: r.v_dots.length,
      warp_per_inch: ww.warp, weft_per_inch: ww.weft,
      iaa: !!r.iaa, blind_mode: !!r.blind_mode, blind_initial: r.blind_initial == null ? null : !!r.blind_initial,
      n_blind_toggles: Array.isArray(r.blind_log) ? r.blind_log.length : null,
      proposals_seen: !!r.proposals_seen, proposal_method: r.proposal_method, proposal_sig: r.proposal_sig,
      n_prop_h: r.n_prop_h, n_prop_v: r.n_prop_v,
      n_prop_kept_h: sum(r.h_orig), n_prop_kept_v: sum(r.v_orig),
      n_user_h: r.h_dots.length - sum(r.h_orig), n_user_v: r.v_dots.length - sum(r.v_orig),
      weave: r.weave, warp_dir: r.warp_dir,
      flags: FLAGS.filter((f) => flags[f.key]).map((f) => f.key).join(';'),
      nonwoven: !!r.nonwoven, confidence: r.confidence, notes: r.notes,
      rect_size: r.rect_size, h_line_y: r.h_line_y, v_line_x: r.v_line_x,
      spacing_est_h: gh ? r1(gh.est) : null, spacing_est_v: gv ? r1(gv.est) : null,
      time_spent_s: Math.round(r.time_spent_s || 0), n_edits: r.n_edits, created: r.created, timestamp: r.timestamp,
      h_dots: r.h_dots.join(' '), v_dots: r.v_dots.join(' '),
      h_orig: (r.h_orig || []).map((o) => (o ? 1 : 0)).join(''), v_orig: (r.v_orig || []).map((o) => (o ? 1 : 0)).join(''),
      warp_tpi: wt.warp == null ? null : r2(wt.warp), weft_tpi: wt.weft == null ? null : r2(wt.weft),
      h_angle_src: angleSource(r, 'h'), v_angle_src: angleSource(r, 'v'), tool_version: r.tool_version,
      view_rot_deg: viewAngle(r),
    };
    Object.assign(row, dv);
    FLAGS.forEach((f) => { row['flag_' + f.key] = !!flags[f.key]; });
    return CSV_COLS.map((k) => csvCell(row[k])).join(',');
  }

  function exportCSV(store) {
    const ids = Object.keys(store.records).map(Number).sort((a, b) => a - b);
    return [CSV_COLS.join(',')].concat(ids.map((i) => csvRow(store.records[i]))).join('\n') + '\n';
  }

  /** RFC-4180 CSV -> array of rows (array of strings). */
  function parseCSV(text) {
    const rows = []; let row = [], cell = '', q = false;
    text = String(text).replace(/^﻿/, '');
    for (let i = 0; i < text.length; i++) {
      const ch = text[i];
      if (q) {
        if (ch === '"' && text[i + 1] === '"') { cell += '"'; i++; } else if (ch === '"') q = false; else cell += ch;
      } else if (ch === '"') q = true;
      else if (ch === ',') { row.push(cell); cell = ''; }
      else if (ch === '\n' || ch === '\r') {
        if (ch === '\r' && text[i + 1] === '\n') i++;
        row.push(cell); rows.push(row); row = []; cell = '';
      } else cell += ch;
    }
    if (cell || row.length) { row.push(cell); rows.push(row); }
    return rows.filter((r) => r.length > 1 || (r.length === 1 && r[0] !== ''));
  }

  /** Records from a CSV written by exportCSV (v1 or v2) -> same shape as parseImport(). */
  function parseImportCSV(text) {
    const rows = parseCSV(text);
    if (rows.length < 1 || rows[0].indexOf('id') < 0 || rows[0].indexOf('h_dots') < 0) {
      throw new Error('not an annotation CSV (needs id and h_dots columns)');
    }
    const H = rows[0];
    const v2 = H.indexOf('h_angle_deg') >= 0 || H.indexOf('v_angle_deg') >= 0;
    const num = (v) => (v === '' || v == null ? null : Number(v));
    const bool = (v) => v === '1' || v === 'true';
    const nums = (v) => (v ? v.trim().split(/\s+/).map(Number).filter((x) => isFinite(x)) : []);
    const bits = (v, n) => (v && v.length === n ? v.split('').map((c) => (c === '1' ? 1 : 0)) : Array(n).fill(0));
    const out = {};
    let annotator = '';
    rows.slice(1).forEach((cells) => {
      const o = {};
      H.forEach((h, i) => { o[h] = cells[i] == null ? '' : cells[i]; });
      if (o.id === '') return;
      const S = num(o.rect_size);
      const h = nums(o.h_dots), v = nums(o.v_dots);
      const flags = emptyFlags();
      FLAGS.forEach((f) => { flags[f.key] = bool(o['flag_' + f.key]); });
      const rec = {
        schema: SCHEMA, id: Number(o.id), annotator: o.annotator || '', rect_size: S,
        h_line_y: num(o.h_line_y), v_line_x: num(o.v_line_x),
        h_dots: h, h_orig: bits(o.h_orig, h.length), v_dots: v, v_orig: bits(o.v_orig, v.length),
        blind_mode: bool(o.blind_mode), blind_initial: o.blind_initial === '' || o.blind_initial == null ? null : bool(o.blind_initial),
        blind_log: [], proposals_seen: bool(o.proposals_seen), iaa: bool(o.iaa),
        proposal_method: o.proposal_method || null, proposal_sig: o.proposal_sig || null,
        n_prop_h: num(o.n_prop_h) || 0, n_prop_v: num(o.n_prop_v) || 0,
        weave: o.weave || null, warp_dir: o.warp_dir || null, flags: flags, nonwoven: bool(o.nonwoven),
        confidence: num(o.confidence), notes: o.notes || '', done: bool(o.done),
        time_spent_s: num(o.time_spent_s) || 0, n_edits: num(o.n_edits),
        created: o.created || null, timestamp: o.timestamp || null, imported_from_csv: true,
      };
      if (rec.n_edits == null) delete rec.n_edits;
      if (v2) {   // v2 CSV: restore the angles, recompute the derived line fields
        ['h', 'v'].forEach((l) => { const a = num(o[l + '_angle_deg']); rec[l + '_angle_deg'] = a == null || !isFinite(a) ? 0 : nz(a); });
        if (o.tool_version) rec.tool_version = o.tool_version;
        const vr = num(o.view_rot_deg);   // v2.2.1 column; absent or 0 -> no field (the default)
        if (vr != null && isFinite(vr) && vr !== 0) rec.view_rot_deg = r2(clamp(vr, -MAX_VIEW, MAX_VIEW));
        updateDerived(rec);
      }
      if (!annotator && rec.annotator) annotator = rec.annotator;
      out[String(rec.id)] = rec;
    });
    return { annotator: annotator, records: out };
  }

  function safeName(s) {
    return String(s || 'anonymous').trim().replace(/[^A-Za-z0-9_-]+/g, '_').slice(0, 40) || 'anonymous';
  }

  return {
    SCHEMA, TOOL_VERSION, APP_VERSION, MAX_ANGLE, MAX_VIEW, WEAVES, WARP_DIRS, FLAGS, CSV_COLS, CSV_COLS_V1, CSV_COLS_V2, CSV_COLS_V221,
    newRecord, insertProposals, removeProposalDots, setBlind, resetToProposals, clearLine,
    proposalSig, isPristine, refreshProposals,
    addDot, removeDot, nearestDot, dotXY, boxDelete, removeOutside, moveLine, counts, tpis, warpWeft, warpWeftTpi,
    gapStats, lineGapStats,
    lineAngle, linePivot, lineDir, pointAt, paramRange, segment, projectToLine, lineInfo, angleSource,
    derived, updateDerived, withDerived, setAngle, rotateBy, keepAudit,
    viewAngle, viewRange, rotateView, setViewAngle, viewAlignDelta, viewToScreen, screenToWorld, viewDir, viewExtent,
    pivotThrough, removeDotsWhere, boxDeleteScreen,
    fftPlan, fft, estimateOrientation,
    median, hashBlind, blindForPolicy, emptyStore, parseImport, parseImportCSV, parseCSV, mergeRecords,
    exportJSON, exportCSV, csvRow, safeName, touch, stamp,
    KIND_LABEL, hasScenes, isCopy, sceneRepId, sceneRotK, copiedLabel, isFlagged, filterIds, progress, syncIaa,
    ATTN_RANK, hasAttention, attentionOf, attentionVisible, attentionDisplay, attentionOrder,
    inIaaScene, sameAnnotator, opensBlind, visibleExamples, finishedScenes, thumbGeometry,
  };
});
