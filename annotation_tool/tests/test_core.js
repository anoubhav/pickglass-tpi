// Headless tests of the annotation logic: `node tests/test_core.js`
// Uses the real data/manifest.json when present (else a synthetic image).
'use strict';
const assert = require('assert');
const fs = require('fs');
const path = require('path');
const C = require('../core.js');

let img = {
  id: 7, rect_size: [500, 500],
  proposal: { h_line_y: 250, v_line_x: 250, h_positions: [10, 30, 50, 70, 90], v_positions: [20, 40, 60], method: 'synthetic' },
};
const mpath = path.join(__dirname, '..', 'data', 'manifest.json');
let manifest = null;
if (fs.existsSync(mpath)) {
  manifest = JSON.parse(fs.readFileSync(mpath, 'utf8'));
  img = manifest.images.find((e) => e.proposal.h_positions.length >= 5) || img;
}
const S = img.rect_size[0], P = img.proposal;
let n = 0;
const t = (name, fn) => { fn(); n++; console.log('ok', n, name); };

t('new record inserts proposals (not blind)', () => {
  const r = C.newRecord(img, { annotator: 'a' });
  assert.strictEqual(r.h_dots.length, P.h_positions.length);
  assert.strictEqual(r.v_dots.length, P.v_positions.length);
  assert.ok(r.proposals_seen && !r.blind_mode);
  assert.ok(r.h_orig.every((o) => o === 1));
  assert.deepStrictEqual(r.h_dots, r.h_dots.slice().sort((a, b) => a - b));
  assert.strictEqual(r.rect_size, S);
});

t('blind record starts empty and never saw proposals', () => {
  const r = C.newRecord(img, { blind: true });
  assert.strictEqual(r.h_dots.length + r.v_dots.length, 0);
  assert.ok(r.blind_mode && !r.proposals_seen);
});

t('add / nearest / remove keep arrays sorted and parallel', () => {
  const r = C.newRecord(img, { blind: true });
  [40, 10, 30, 20].forEach((x) => C.addDot(r, 'h', x));
  assert.deepStrictEqual(r.h_dots, [10, 20, 30, 40]);
  assert.deepStrictEqual(r.h_orig, [0, 0, 0, 0]);
  const nd = C.nearestDot(r, 'h', 21);
  assert.strictEqual(nd.index, 1);
  C.removeDot(r, 'h', nd.index);
  assert.deepStrictEqual(r.h_dots, [10, 30, 40]);
  C.addDot(r, 'h', -5); C.addDot(r, 'h', S + 9);
  assert.strictEqual(r.h_dots[0], 0);
  assert.strictEqual(r.h_dots[r.h_dots.length - 1], S);
  assert.strictEqual(C.counts(r).h, 5);
});

t('blind toggle removes only proposal dots; un-blind restores without duplicates', () => {
  const r = C.newRecord(img, {});
  const before = r.h_dots.length;
  C.addDot(r, 'h', 1.0);
  C.setBlind(r, img, true);
  assert.deepStrictEqual(r.h_dots, [1]);
  assert.strictEqual(r.v_dots.length, 0);
  C.setBlind(r, img, false);
  // proposal within 1.5 px of the user dot would be skipped, otherwise all come back
  const near = P.h_positions.filter((x) => Math.abs(x - 1) < 1.5).length;
  assert.strictEqual(r.h_dots.length, before + 1 - near);
  assert.ok(r.proposals_seen && !r.blind_mode);
});

t('box delete removes dots on both lines inside the box only', () => {
  const r = C.newRecord(img, { blind: true });
  C.moveLine(r, 'h', 100); C.moveLine(r, 'v', 200);
  [50, 150, 250].forEach((x) => C.addDot(r, 'h', x));   // at (x, 100)
  [50, 150, 250].forEach((y) => C.addDot(r, 'v', y));   // at (200, y)
  const k = C.boxDelete(r, 120, 80, 260, 160);          // contains (150,100), (250,100), (200,150)
  assert.strictEqual(k, 3);
  assert.deepStrictEqual(r.h_dots, [50]);
  assert.deepStrictEqual(r.v_dots, [50, 250]);
});

t('moving a line keeps dot positions along it; clamps to window', () => {
  const r = C.newRecord(img, {});
  const d = r.h_dots.slice();
  C.moveLine(r, 'h', -40);
  assert.strictEqual(r.h_line_y, 0);
  assert.deepStrictEqual(r.h_dots, d);
});

t('warp/weft mapping follows warp direction', () => {
  const r = C.newRecord(img, { blind: true });
  C.addDot(r, 'h', 10); C.addDot(r, 'h', 20); C.addDot(r, 'v', 5);
  assert.deepStrictEqual(C.warpWeft(r), { warp: null, weft: null });
  r.warp_dir = 'vertical';
  assert.deepStrictEqual(C.warpWeft(r), { warp: 2, weft: 1 });
  r.warp_dir = 'horizontal';
  assert.deepStrictEqual(C.warpWeft(r), { warp: 1, weft: 2 });
});

t('gap stats flag a missing yarn and a duplicate', () => {
  const dots = [];
  for (let x = 5; x < 100; x += 10) if (x !== 45) dots.push(x);
  dots.push(76); dots.sort((a, b) => a - b);
  const g = C.gapStats(dots, 100);
  assert.ok(Math.abs(g.median - 10) < 1e-9 || g.median > 0);
  assert.ok(g.big.some(([a, b]) => a === 35 && b === 55), 'missing yarn at 45');
  assert.ok(g.small.length >= 1, 'duplicate near 75/76');
  assert.strictEqual(C.gapStats([1, 2], 100), null);
});

t('reset to proposals restores the proposal set', () => {
  const r = C.newRecord(img, {});
  C.clearLine(r, 'h'); C.addDot(r, 'v', 3);
  C.resetToProposals(r, img);
  assert.strictEqual(r.h_dots.length, P.h_positions.length);
  assert.strictEqual(r.v_dots.length, P.v_positions.length);
});

t('CSV export: header, one row per record, escaping', () => {
  const store = C.emptyStore('alice');
  const r = C.newRecord(img, { annotator: 'alice' });
  r.notes = 'glare, "hot" spot\nleft side'; r.weave = 'twill_2_1'; r.warp_dir = 'vertical';
  r.flags.glare = true; r.confidence = 2; r.done = true;
  store.records[String(r.id)] = r;
  const csv = C.exportCSV(store);
  const lines = csv.trim().split('\n');
  assert.strictEqual(lines[0], C.CSV_COLS.join(','));
  assert.ok(csv.includes('"glare, ""hot"" spot\nleft side"'));
  // parse the row back with a tiny CSV parser
  const rows = parseCSV(csv);
  assert.strictEqual(rows.length, 2);
  const row = Object.fromEntries(rows[0].map((h, i) => [h, rows[1][i]]));
  assert.strictEqual(Number(row.h_count), r.h_dots.length);
  assert.strictEqual(Number(row.warp_per_inch), r.h_dots.length);
  assert.strictEqual(row.flags, 'glare');
  assert.strictEqual(row.flag_glare, '1');
  assert.strictEqual(row.weave, 'twill_2_1');
  assert.strictEqual(row.h_dots.split(' ').length, r.h_dots.length);
});

t('JSON export -> import round trip and newest-wins merge', () => {
  const store = C.emptyStore('bob');
  const a = C.newRecord(img, { annotator: 'bob', now: '2026-10-01T10:00:00.000Z' });
  a.timestamp = '2026-10-01T10:00:00.000Z';
  store.records[String(a.id)] = a;
  const exported = JSON.parse(JSON.stringify(C.exportJSON(store, manifest)));
  const parsed = C.parseImport(exported);
  assert.strictEqual(parsed.annotator, 'bob');
  assert.deepStrictEqual(parsed.records[String(a.id)], a);
  const newer = JSON.parse(JSON.stringify(a)); newer.timestamp = '2026-10-02T10:00:00.000Z'; newer.weave = 'satin';
  const other = JSON.parse(JSON.stringify(a)); other.id = 9999;
  const m = C.mergeRecords(store.records, { [a.id]: newer, 9999: other });
  assert.strictEqual(m.added, 1); assert.strictEqual(m.updated, 1);
  assert.strictEqual(m.merged[String(a.id)].weave, 'satin');
  const m2 = C.mergeRecords(m.merged, { [a.id]: a });
  assert.strictEqual(m2.kept, 1);
  assert.strictEqual(m2.merged[String(a.id)].weave, 'satin');
  assert.throws(() => C.parseImport({ foo: 1 }));
});

t('merge: an untouched record never replaces an edited one, whatever the timestamps', () => {
  const edited = C.newRecord(img, { now: '2026-10-01T10:00:00.000Z' });
  C.addDot(edited, 'h', 3); edited.done = true; edited.timestamp = '2026-10-01T10:00:00.000Z';
  const fresh = C.newRecord(img, {});  // newer timestamp, n_edits 0
  assert.ok(Date.parse(fresh.timestamp) > Date.parse(edited.timestamp));
  let m = C.mergeRecords({ [img.id]: edited }, { [img.id]: fresh });
  assert.strictEqual(m.merged[img.id], edited); assert.strictEqual(m.kept, 1);
  m = C.mergeRecords({ [img.id]: fresh }, { [img.id]: edited });
  assert.strictEqual(m.merged[img.id], edited); assert.strictEqual(m.updated, 1);
  const legacy = JSON.parse(JSON.stringify(fresh)); delete legacy.n_edits; legacy.timestamp = '2020-01-01T00:00:00.000Z';
  m = C.mergeRecords({ [img.id]: fresh }, { [img.id]: legacy });
  assert.strictEqual(m.merged[img.id], legacy, 'legacy records count as edited');
});

t('blind policies', () => {
  assert.strictEqual(C.blindForPolicy('all', 3, false), true);
  assert.strictEqual(C.blindForPolicy('none', 3, true), false);
  assert.strictEqual(C.blindForPolicy('manual', 3, true), true);
  const k = Array.from({ length: 150 }, (_, i) => C.hashBlind(i)).filter(Boolean).length;
  assert.ok(k > 55 && k < 95, 'random policy is roughly 50%: ' + k);
});

t('edits are counted; creation is not an edit; pristine records adopt refreshed proposals', () => {
  const r = C.newRecord(img, { annotator: 'a', iaa: true });
  assert.strictEqual(r.n_edits, 0);
  assert.ok(C.isPristine(r) && r.iaa && r.proposal_sig === C.proposalSig(img));
  assert.strictEqual(C.refreshProposals(r, img), 'same');
  const img2 = JSON.parse(JSON.stringify(img));
  img2.proposal.method = 'ours'; img2.proposal.h_positions = [11, 22, 33]; img2.proposal.v_positions = [44];
  const created = r.created; r.time_spent_s = 12;
  assert.notStrictEqual(C.proposalSig(img2), C.proposalSig(img));
  assert.strictEqual(C.refreshProposals(r, img2), 'refreshed');
  assert.deepStrictEqual(r.h_dots, [11, 22, 33]);
  assert.deepStrictEqual(r.v_dots, [44]);
  assert.strictEqual(r.proposal_method, 'ours');
  assert.strictEqual(r.created, created); assert.strictEqual(r.time_spent_s, 12);
  assert.ok(r.iaa && r.n_edits === 0);
  // one user edit -> never overwritten again
  C.addDot(r, 'h', 50);
  assert.strictEqual(r.n_edits, 1);
  assert.strictEqual(C.refreshProposals(r, img), 'stale');
  assert.deepStrictEqual(r.h_dots, [11, 22, 33, 50]);
  // explicit reset switches and records the new signature
  C.resetToProposals(r, img);
  assert.strictEqual(r.proposal_sig, C.proposalSig(img));
  assert.strictEqual(r.h_dots.length, P.h_positions.length);
  // done records are never pristine; legacy records (no n_edits) neither
  const d = C.newRecord(img, {}); d.done = true; assert.ok(!C.isPristine(d));
  const l = C.newRecord(img, {}); delete l.n_edits; assert.ok(!C.isPristine(l));
});

t('blind state is logged; blind-initial records never saw proposals', () => {
  const r = C.newRecord(img, { blind: true });
  assert.ok(r.blind_initial && r.blind_mode && !r.proposals_seen && r.blind_log.length === 0);
  C.setBlind(r, img, false);
  C.setBlind(r, img, true);
  assert.strictEqual(r.blind_log.length, 2);
  assert.deepStrictEqual(r.blind_log.map((e) => e.on), [false, true]);
  assert.ok(r.blind_mode && r.proposals_seen && r.blind_initial, 'blind again but contaminated');
  C.setBlind(r, img, true); // no-op
  assert.strictEqual(r.blind_log.length, 2);
});

t('un-blinding after a blind count skips proposals that duplicate the user dots', () => {
  const im = { id: 3, rect_size: [200, 200],
    proposal: { h_line_y: 100, v_line_x: 100, h_positions: [12, 22, 31, 41, 50, 60, 70, 80], v_positions: [], method: 'x' } };
  const r = C.newRecord(im, { blind: true });
  [10, 20, 30, 40, 50].forEach((x) => C.addDot(r, 'h', x));  // user counted the first 5 yarns
  C.setBlind(r, im, false);
  // 12/22/31/41/50 are within 0.4*10 px of user dots -> skipped; 60/70/80 inserted
  assert.deepStrictEqual(r.h_dots, [10, 20, 30, 40, 50, 60, 70, 80]);
  assert.deepStrictEqual(r.h_orig, [0, 0, 0, 0, 0, 1, 1, 1]);
  assert.strictEqual(r.n_prop_h, 3);
});

t('IAA blind policy', () => {
  assert.strictEqual(C.blindForPolicy('iaa', 3, false, true), true);
  assert.strictEqual(C.blindForPolicy('iaa', 3, true, false), false);
});

t('CSV export -> CSV import round trip keeps counts, origins, flags and fields', () => {
  const store = C.emptyStore('carol');
  const r = C.newRecord(img, { annotator: 'carol', iaa: true });
  C.addDot(r, 'h', 1.5); r.weave = 'satin'; r.warp_dir = 'horizontal'; r.flags.blur = true;
  r.confidence = 3; r.notes = 'a, "b"\nc'; r.done = true; r.time_spent_s = 33;
  const r2 = C.newRecord(Object.assign({}, img, { id: img.id + 1 }), { annotator: 'carol', blind: true });
  store.records[String(r.id)] = r; store.records[String(r2.id)] = r2;
  const back = C.parseImportCSV(C.exportCSV(store));
  assert.strictEqual(back.annotator, 'carol');
  const q = back.records[String(r.id)];
  ['h_dots', 'v_dots', 'h_orig', 'v_orig', 'weave', 'warp_dir', 'confidence', 'notes', 'done', 'iaa',
    'blind_mode', 'blind_initial', 'proposals_seen', 'rect_size', 'h_line_y', 'v_line_x', 'n_edits',
    'proposal_sig', 'timestamp', 'created'].forEach((k) => assert.deepStrictEqual(q[k], r[k], k));
  assert.strictEqual(q.flags.blur, true); assert.strictEqual(q.flags.glare, false);
  assert.strictEqual(q.time_spent_s, 33);
  const q2 = back.records[String(r2.id)];
  assert.ok(q2.blind_mode && !q2.proposals_seen && q2.h_dots.length === 0);
  // CRLF + BOM tolerant, and rejects random CSVs
  const crlf = '\uFEFF' + C.exportCSV(store).replace(/\n/g, '\r\n');
  assert.strictEqual(Object.keys(C.parseImportCSV(crlf).records).length, 2);
  assert.throws(() => C.parseImportCSV('a,b\n1,2\n'));
});

// ------------------------------------------------------------------ v2: rotatable guide lines
const near = (a, b, eps, msg) => assert.ok(Math.abs(a - b) <= (eps || 1e-9), (msg || '') + ' ' + a + ' vs ' + b);
// a record exactly as v1 wrote it (copied from a real v1 autosave): no angle / derived fields
const V1_REC = {
  schema: 1, id: 0, annotator: '', rect_size: 490, h_line_y: 245, v_line_x: 305,
  h_dots: [109.8, 163.3, 206.7, 309.1], h_orig: [0, 0, 0, 0], v_dots: [168.7, 201, 224.8, 388], v_orig: [0, 0, 0, 0],
  blind_mode: true, blind_initial: false, blind_log: [{ t: '2026-10-07T18:50:13.794Z', on: true }], proposals_seen: true,
  iaa: false, proposal_method: 'ours-2026-lattice-v1', proposal_sig: 'ours-2026-lattice-v1:1d6237b4', n_prop_h: 47, n_prop_v: 39,
  weave: 'twill_2_1', warp_dir: null,
  flags: { glare: false, blur: false, shadow: false, low_res: false, perspective: false, occluded: false },
  nonwoven: false, confidence: null, notes: '', done: false, time_spent_s: 143, n_edits: 149, tool_version: '2026-10-07',
  created: '2026-10-07T18:48:37.472Z', timestamp: '2026-10-07T18:50:55.907Z',
};
const V1_CSV_HEADER = 'id,annotator,done,h_count,v_count,n_dots_h,n_dots_v,warp_per_inch,weft_per_inch,iaa,blind_mode,' +
  'blind_initial,n_blind_toggles,proposals_seen,proposal_method,proposal_sig,n_prop_h,n_prop_v,n_prop_kept_h,n_prop_kept_v,' +
  'n_user_h,n_user_v,weave,warp_dir,flags,flag_glare,flag_blur,flag_shadow,flag_low_res,flag_perspective,flag_occluded,' +
  'nonwoven,confidence,notes,rect_size,h_line_y,v_line_x,spacing_est_h,spacing_est_v,time_spent_s,n_edits,created,timestamp,' +
  'h_dots,v_dots,h_orig,v_orig';
const clone = (o) => JSON.parse(JSON.stringify(o));

t('v1 record: angle 0 geometry is exactly the v1 geometry (dots, segment, TPI == count)', () => {
  const r = clone(V1_REC);
  assert.strictEqual(C.lineAngle(r, 'h'), 0); assert.strictEqual(C.lineAngle(r, 'v'), 0);
  r.h_dots.forEach((x, i) => assert.deepStrictEqual(C.dotXY(r, 'h', i), [x, r.h_line_y]));
  r.v_dots.forEach((y, i) => assert.deepStrictEqual(C.dotXY(r, 'v', i), [r.v_line_x, y]));
  assert.deepStrictEqual(C.segment(r, 'h'), [[0, 245], [490, 245]]);
  assert.deepStrictEqual(C.segment(r, 'v'), [[305, 0], [305, 490]]);
  ['h', 'v'].forEach((l) => {
    const L = C.lineInfo(r, l);
    assert.strictEqual(L.segPx, 490); assert.strictEqual(L.segIn, 1); assert.strictEqual(L.tpi, r[l + '_dots'].length);
  });
  assert.deepStrictEqual(C.tpis(r), C.counts(r));
  assert.deepStrictEqual(r, V1_REC, 'reading geometry never mutates a v1 record');
  // identity holds for every line position, including the window edges
  [0, 0.1, 17.3, 245, 489.9, 490].forEach((y) => {
    const q = clone(V1_REC); q.h_line_y = y; q.v_line_x = y;
    assert.strictEqual(C.lineInfo(q, 'h').tpi, 4); assert.strictEqual(C.lineInfo(q, 'v').segIn, 1);
  });
  const d = C.derived(r);
  assert.deepStrictEqual(d, { h_angle_deg: 0, h_seg_len_px: 490, h_seg_len_in: 1, h_raw_count: 4, h_tpi: 4,
    v_angle_deg: 0, v_seg_len_px: 490, v_seg_len_in: 1, v_raw_count: 4, v_tpi: 4 });
});

t('rotation geometry: sign convention, pivot, perpendicular lines, clipping to the window', () => {
  const r = C.newRecord(img, { blind: true });
  const S = r.rect_size;
  C.moveLine(r, 'h', S / 2); C.moveLine(r, 'v', S / 2);
  C.setAngle(r, 'h', 10, 'key'); C.setAngle(r, 'v', 10, 'key');
  // +angle = clockwise on screen (y down): the H line's right end goes DOWN, the V line's bottom end goes LEFT
  const sh = C.segment(r, 'h'), sv = C.segment(r, 'v');
  assert.ok(sh[1][1] > sh[0][1], 'H right end lower');
  assert.ok(sv[1][0] < sv[0][0], 'V bottom end left');
  // the pivot stays on the line, and equal angles keep the lines perpendicular
  const pH = C.pointAt(r, 'h', S / 2); near(pH[1], r.h_line_y, 1e-9, 'pivot');
  const uh = C.lineDir(r, 'h'), uv = C.lineDir(r, 'v');
  near(uh[0] * uv[0] + uh[1] * uv[1], 0, 1e-12, 'perpendicular');
  near(Math.hypot(uh[0], uh[1]), 1, 1e-12); near(Math.atan2(uh[1], uh[0]) * 180 / Math.PI, 10, 1e-9);
  // through the centre at 10 deg the H line leaves through the left/right edges: length S / cos 10
  const L = C.lineInfo(r, 'h');
  near(L.lo, 0); near(L.hi, S); near(L.segPx, S / Math.cos(10 * Math.PI / 180), 1e-6);
  // near the top edge the segment is clipped by y = 0, so it is shorter and starts at x > 0
  C.moveLine(r, 'h', 5);
  const L2 = C.lineInfo(r, 'h'), seg2 = C.segment(r, 'h');
  near(seg2[0][1], 0, 1e-9, 'starts on the top edge'); assert.ok(L2.lo > 0 && L2.hi === S && L2.segPx < L.segPx);
  seg2.forEach(([x, y]) => assert.ok(x >= -1e-9 && x <= S + 1e-9 && y >= -1e-9 && y <= S + 1e-9));
  // angle is clamped to +-45 and invalid values read as 0
  C.setAngle(r, 'v', 80, 'key'); assert.strictEqual(r.v_angle_deg, C.MAX_ANGLE);
  C.setAngle(r, 'v', -1e9, 'key'); assert.strictEqual(r.v_angle_deg, -C.MAX_ANGLE);
  assert.strictEqual(C.lineAngle({ h_angle_deg: 'x' }, 'h'), 0);
  assert.strictEqual(C.lineAngle({ h_angle_deg: null }, 'h'), 0);
});

t('normalised TPI: 45 deg through the centre = dots / sqrt(2); clipped segments use their own length', () => {
  const r = C.newRecord(img, { blind: true }), S = r.rect_size;
  C.moveLine(r, 'h', S / 2);
  C.setAngle(r, 'h', 45, 'key');
  for (let i = 0; i < 20; i++) C.addDot(r, 'h', S * (i + 0.5) / 20);
  const L = C.lineInfo(r, 'h');
  near(L.segPx, S * Math.SQRT2, 1e-6); near(L.segIn, Math.SQRT2, 1e-9);
  near(L.tpi, 20 / Math.SQRT2, 1e-9); assert.strictEqual(L.n, 20);
  assert.strictEqual(r.h_raw_count, 20); near(r.h_tpi, 14.14, 1e-9); near(r.h_seg_len_in, 1.4142, 1e-9);
  // -45 deg through a point near the bottom: segment from the left edge to the bottom edge
  C.moveLine(r, 'h', S * 0.9); C.setAngle(r, 'h', -45, 'key');
  const seg = C.segment(r, 'h'), L2 = C.lineInfo(r, 'h');
  near(Math.hypot(seg[1][0] - seg[0][0], seg[1][1] - seg[0][1]), L2.segPx, 1e-6);
  near(L2.tpi, L2.nIn / L2.segIn, 1e-9);
  // small angle: 10 deg -> 1/cos(10) inch, 47 dots -> 46.29 TPI
  const q = C.newRecord(img, { blind: true }); C.moveLine(q, 'v', q.rect_size / 2);
  for (let i = 0; i < 47; i++) C.addDot(q, 'v', q.rect_size * (i + 0.5) / 47);
  C.setAngle(q, 'v', -10, 'auto');
  near(C.lineInfo(q, 'v').tpi, 47 * Math.cos(10 * Math.PI / 180), 1e-9);
  assert.strictEqual(C.counts(q).v, 47, 'raw count unchanged by rotation');
});

t('clicks are projected onto the rotated line; dots keep their parameter when the line turns', () => {
  const r = C.newRecord(img, { blind: true }), S = r.rect_size;
  C.moveLine(r, 'h', S / 2); C.setAngle(r, 'h', 12, 'key');
  const p = 0.3 * S, on = C.pointAt(r, 'h', p), u = C.lineDir(r, 'h'), n = [-u[1], u[0]];
  // a click 4 px off the line, perpendicular to it, projects back to the same parameter
  const pr = C.projectToLine(r, 'h', on[0] + 4 * n[0], on[1] + 4 * n[1]);
  near(pr.pos, p, 1e-9); near(pr.dist, 4, 1e-9); assert.ok(pr.inside);
  const i = C.addDot(r, 'h', pr.pos);
  const xy = C.dotXY(r, 'h', i); near(C.projectToLine(r, 'h', xy[0], xy[1]).dist, 0, 1e-9, 'dot on the line');
  // a click beyond the window end is clamped onto the segment
  const far = C.pointAt(r, 'h', S + 40);
  const pf = C.projectToLine(r, 'h', far[0], far[1]);
  assert.ok(!pf.inside); near(pf.pos, C.paramRange(r, 'h').hi);
  // V line: same, with y as the parameter
  C.moveLine(r, 'v', S / 2); C.setAngle(r, 'v', -7, 'key');
  const qv = C.pointAt(r, 'v', 0.6 * S), uv = C.lineDir(r, 'v');
  const pv = C.projectToLine(r, 'v', qv[0] + 3 * uv[1], qv[1] - 3 * uv[0]);
  near(pv.pos, 0.6 * S, 1e-9); near(pv.dist, 3, 1e-9);
  // rotating keeps each dot's x (H) and it stays on the line; rotating back restores the old position
  const before = r.h_dots.slice(), xy0 = C.dotXY(r, 'h', 0);
  C.setAngle(r, 'h', -20, 'drag');
  assert.deepStrictEqual(r.h_dots, before);
  const xy1 = C.dotXY(r, 'h', 0); near(C.projectToLine(r, 'h', xy1[0], xy1[1]).dist, 0, 1e-9);
  C.setAngle(r, 'h', 12, 'drag'); assert.deepStrictEqual(C.dotXY(r, 'h', 0), xy0);
  // box delete works on rotated dot positions
  const k = C.boxDelete(r, xy0[0] - 1, xy0[1] - 1, xy0[0] + 1, xy0[1] + 1);
  assert.strictEqual(k, 1); assert.strictEqual(r.h_dots.length, 0);
});

t('setAngle: no-op does not touch, every change is logged (key bursts coalesced), reset levels lines', () => {
  const r = clone(V1_REC);
  assert.strictEqual(C.setAngle(r, 'h', 0, 'zero'), false);
  assert.deepStrictEqual(r, V1_REC, 'pressing 0 on a level v1 line changes nothing');
  assert.ok(C.rotateBy(r, 'h', 0.5));
  assert.ok(C.rotateBy(r, 'h', 0.5));
  assert.strictEqual(r.h_angle_deg, 1); assert.strictEqual(r.n_edits, 151);
  assert.strictEqual(r.angle_log.length, 1, 'burst of key presses = one entry');
  assert.deepStrictEqual([r.angle_log[0].line, r.angle_log[0].how, r.angle_log[0].deg], ['h', 'key', 1]);
  C.setAngle(r, 'v', -3.333, 'auto', { est: -3.333, clear: true });
  assert.strictEqual(r.v_angle_deg, -3.33);
  assert.strictEqual(C.angleSource(r, 'v'), 'auto'); assert.strictEqual(C.angleSource(r, 'h'), 'key');
  assert.strictEqual(r.angle_log.length, 2);
  near(r.h_tpi, Math.round(100 * 4 * Math.cos(Math.PI / 180)) / 100, 1e-9, 'derived fields refreshed on every edit');
  const im = { id: 0, rect_size: [490, 490], proposal: { h_line_y: 245, v_line_x: 245, h_positions: [10, 20], v_positions: [30], method: 'm' } };
  r.blind_mode = false;
  C.resetToProposals(r, im);
  assert.strictEqual(r.h_angle_deg, 0); assert.strictEqual(r.v_angle_deg, 0);
  assert.deepStrictEqual(r.h_dots, [10, 20]);
  assert.strictEqual(C.angleSource(r, 'h'), 'reset');
});

t('dots outside the window after rotating: excluded from TPI, flagged, removable', () => {
  const r = C.newRecord(img, { blind: true }), S = r.rect_size;
  C.moveLine(r, 'h', S / 2);
  [0.05, 0.5, 0.95].forEach((f) => C.addDot(r, 'h', f * S));
  C.moveLine(r, 'h', 0.05 * S);
  C.setAngle(r, 'h', 30, 'key');   // pivot near the top: the left part of the line leaves the window
  const L = C.lineInfo(r, 'h');
  assert.strictEqual(L.n, 3); assert.strictEqual(L.nOut, 1); assert.strictEqual(L.nIn, 2);
  near(L.tpi, 2 / L.segIn, 1e-9);
  assert.strictEqual(r.h_raw_count, 3);
  assert.strictEqual(C.removeOutside(r, 'h'), 1);
  assert.strictEqual(r.h_dots.length, 2); assert.strictEqual(C.lineInfo(r, 'h').nOut, 0);
  // proposals are only inserted where the rotated segment is
  const im = { id: 1, rect_size: [S, S], proposal: { h_line_y: S / 2, v_line_x: S / 2, h_positions: [1, S / 2, S - 1], v_positions: [], method: 'x' } };
  const q = C.newRecord(im, { blind: true }); C.moveLine(q, 'h', 2); C.setAngle(q, 'h', 20, 'key');
  C.insertProposals(q, im, 0.5);
  q.h_dots.forEach((x) => assert.ok(x >= C.paramRange(q, 'h').lo - 0.05));
});

t('gap statistics on a rotated line use the segment ends and the along-line spacing', () => {
  const dots = [10, 20, 30, 40, 50, 60, 70, 80, 90];
  assert.deepStrictEqual(C.gapStats(dots, 100), C.gapStats(dots, 100, 0, 100, 1), 'defaults = v1');
  const g = C.gapStats(dots, 100, 0, 100, 1 / Math.cos(Math.PI / 6));
  near(g.est, 10 * Math.cos(Math.PI / 6), 1e-9);
  const g2 = C.gapStats(dots, 100, 9, 91);
  assert.strictEqual(g2.big.length, 0, 'no edge gaps inside a shorter segment');
});

t('JSON round trip with and without angle fields; export never mutates the store', () => {
  const store = C.emptyStore('');
  store.records['0'] = clone(V1_REC);
  const q = C.newRecord(Object.assign({}, img, { id: 5 }), { annotator: '' });
  C.setAngle(q, 'h', -10.68, 'auto', { est: -10.68, clear: true });
  store.records['5'] = q;
  const ex = JSON.parse(JSON.stringify(C.exportJSON(store, manifest)));
  assert.deepStrictEqual(store.records['0'], V1_REC, 'store record untouched by export');
  assert.strictEqual(ex.tool_version, C.TOOL_VERSION);
  const e0 = ex.records.find((r) => r.id === 0), e5 = ex.records.find((r) => r.id === 5);
  // exported v1 record = v1 fields unchanged + derived fields (angle 0, TPI == count)
  Object.keys(V1_REC).forEach((k) => assert.deepStrictEqual(e0[k], V1_REC[k], k));
  assert.strictEqual(e0.h_tpi, 4); assert.strictEqual(e0.h_angle_deg, 0); assert.strictEqual(e0.v_seg_len_in, 1);
  assert.strictEqual(e5.h_angle_deg, -10.68); near(e5.h_tpi, Math.round(100 * q.h_dots.length * Math.cos(10.68 * Math.PI / 180)) / 100, 1e-9);
  const back = C.parseImport(ex);
  assert.deepStrictEqual(back.records['5'], C.withDerived(q));
  assert.strictEqual(C.lineAngle(back.records['5'], 'h'), -10.68);
  // a v1 file (no angle fields) imports unchanged
  const v1 = C.parseImport({ annotator: '', records: [clone(V1_REC)] });
  assert.deepStrictEqual(v1.records['0'], V1_REC);
});

t('merge: old (v1) and new (v2) records of the same image, newest wins either way', () => {
  const old = clone(V1_REC);
  const neu = clone(V1_REC); C.setAngle(neu, 'h', 4, 'key');   // touch(): newer timestamp
  let m = C.mergeRecords({ 0: old }, { 0: neu });
  assert.strictEqual(m.updated, 1); assert.strictEqual(m.merged[0].h_angle_deg, 4);
  m = C.mergeRecords({ 0: neu }, { 0: old });
  assert.strictEqual(m.kept, 1); assert.strictEqual(m.merged[0], neu);
  const newerOld = clone(V1_REC); newerOld.timestamp = new Date(Date.now() + 3600e3).toISOString();
  m = C.mergeRecords({ 0: neu }, { 0: newerOld });
  assert.strictEqual(m.merged[0], newerOld); assert.strictEqual(C.lineInfo(m.merged[0], 'h').tpi, 4, 'v1 record wins -> level line, TPI == count');
});

t('CSV: v1 columns unchanged and first, v2 columns appended; v1 and v2 CSVs import', () => {
  assert.strictEqual(C.CSV_COLS_V1.join(','), V1_CSV_HEADER, 'v1 header byte-identical');
  assert.deepStrictEqual(C.CSV_COLS.slice(0, C.CSV_COLS_V1.length), C.CSV_COLS_V1);
  assert.deepStrictEqual(C.CSV_COLS.slice(C.CSV_COLS_V1.length, C.CSV_COLS_V1.length + 15), ['h_angle_deg', 'v_angle_deg', 'h_seg_len_px', 'v_seg_len_px',
    'h_seg_len_in', 'v_seg_len_in', 'h_raw_count', 'v_raw_count', 'h_tpi', 'v_tpi', 'warp_tpi', 'weft_tpi',
    'h_angle_src', 'v_angle_src', 'tool_version']);
  // v2.2.1 appends view_rot_deg at the very end (checked in detail in the view-turn tests below)
  assert.deepStrictEqual(C.CSV_COLS.slice(C.CSV_COLS_V1.length + 15), ['view_rot_deg']);
  const store = C.emptyStore('dana');
  const a = clone(V1_REC); a.annotator = 'dana'; a.warp_dir = 'vertical'; store.records['0'] = a;
  const b = C.newRecord(Object.assign({}, img, { id: 3 }), { annotator: 'dana' });
  b.warp_dir = 'horizontal'; C.setAngle(b, 'v', 8, 'drag'); store.records['3'] = b;
  const csv = C.exportCSV(store);
  const rows = C.parseCSV(csv), H = rows[0];
  const row = (id) => Object.fromEntries(H.map((h, i) => [h, rows.find((x) => x[0] === String(id))[i]]));
  const r0 = row(0), r3 = row(3);
  // v1 columns of a v1 record are exactly what v1 wrote (prefix of the line)
  assert.strictEqual(r0.h_count, '4'); assert.strictEqual(r0.h_tpi, '4'); assert.strictEqual(r0.h_angle_deg, '0');
  assert.strictEqual(r0.warp_tpi, '4'); assert.strictEqual(r0.h_angle_src, ''); assert.strictEqual(r0.tool_version, '2026-10-07');
  assert.strictEqual(r3.v_angle_deg, '8'); assert.strictEqual(r3.v_angle_src, 'drag');
  assert.strictEqual(Number(r3.v_count), b.v_dots.length, 'v_count stays the raw count');
  near(Number(r3.v_tpi), Math.round(100 * b.v_dots.length * Math.cos(8 * Math.PI / 180)) / 100, 1e-9);
  assert.strictEqual(r3.warp_tpi, r3.v_tpi);
  // v2 CSV -> records with angles; v1 CSV (v1 columns only) -> records without angle fields
  const back = C.parseImportCSV(csv);
  assert.strictEqual(back.records['3'].v_angle_deg, 8); assert.strictEqual(back.records['3'].v_tpi, b.v_tpi);
  const v1csv = rows.map((r) => r.slice(0, C.CSV_COLS_V1.length).map((c) => (/[",\n]/.test(c) ? '"' + c.replace(/"/g, '""') + '"' : c)).join(',')).join('\n');
  const old = C.parseImportCSV(v1csv);
  assert.ok(!('h_angle_deg' in old.records['3']) && !('h_tpi' in old.records['3']));
  assert.strictEqual(C.lineInfo(old.records['0'], 'h').tpi, 4);
});

t('orientation estimate recovers synthetic yarn angles (both families, twill harmonics, noise)', () => {
  const N = 512;   // the app's resolution (Nyquist 256 cycles per inch)
  function fabric(fh, ah, fv, av, twill, seed) {
    let s = seed; const rnd = () => { s = (s * 16807) % 2147483647; return s / 2147483647; };
    const g = new Float32Array(N * N), a = ah * Math.PI / 180, b = av * Math.PI / 180;
    for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
      const p1 = 2 * Math.PI * fh * (x * Math.cos(a) + y * Math.sin(a)) / N;
      const p2 = 2 * Math.PI * fv * (-x * Math.sin(b) + y * Math.cos(b)) / N;
      let v = Math.abs(Math.cos(p1 / 2)) ** 3 * (Math.cos(p2) > 0 ? 1 : 0.6) + Math.abs(Math.cos(p2 / 2)) ** 3 * (Math.cos(p1) > 0 ? 0.6 : 1);
      if (twill) v += 0.5 * Math.cos((p1 + p2) / 1.5);
      g[y * N + x] = 100 + 60 * v + 40 * (rnd() - 0.5) + 20 * Math.sin(2 * Math.PI * x / N);
    }
    return g;
  }
  [[47, -10.5, 38, -12.8, 0], [40, 0, 40, 0, 1], [120, 3.3, 90, 2.1, 1], [25, 7, 30, 4, 0]].forEach(([fh, ah, fv, av, tw], k) => {
    const e = C.estimateOrientation(fabric(fh, ah, fv, av, tw, 11 + k), N, {});
    near(e.h.deg, ah, 0.3, 'H'); near(e.v.deg, av, 0.3, 'V');
    assert.ok(e.h.clear && e.v.clear, 'clear estimate');
  });
  // an implausible H/V skew (> 8 deg) is never reported as clear
  const sk = C.estimateOrientation(fabric(50, 9, 50, -6, 0, 3), N, { skew: 0 });
  assert.ok(Math.abs(sk.h.deg - 9) < 0.5 && Math.abs(sk.v.deg + 6) < 0.5);
  assert.ok(!sk.h.clear && !sk.v.clear && sk.h.skewed);
  // featureless image: no direction
  const flat = C.estimateOrientation(new Float32Array(N * N).fill(7), N, {});
  assert.strictEqual(flat.h, null); assert.strictEqual(flat.v, null);
  assert.throws(() => C.fftPlan(100));
});

t('undo keeps the audit trail: proposals_seen sticky, blind/angle logs append-only, time never lost', () => {
  // the app's undo: snapshot = JSON of the record before an edit; restore = keepAudit(snapshot, current)
  const r = C.newRecord(img, { blind: true });
  C.addDot(r, 'h', 12.5);
  const snapBlind = JSON.stringify(r);
  C.setBlind(r, img, false);                       // proposals now on screen
  assert.ok(r.proposals_seen && r.blind_log.length === 1);
  r.time_spent_s = 90;                             // active time credited after the snapshot
  const u1 = C.keepAudit(JSON.parse(snapBlind), r);
  assert.strictEqual(u1.blind_mode, true);
  assert.deepStrictEqual(u1.h_dots, [12.5]);       // the edit itself is undone
  assert.strictEqual(u1.proposals_seen, true);     // ... but the proposals were seen
  assert.strictEqual(u1.time_spent_s, 90);
  assert.strictEqual(u1.blind_log.length, 2);
  assert.deepStrictEqual([u1.blind_log[0].on, u1.blind_log[1].on, u1.blind_log[1].undo], [false, true, true]);
  // angles: the undo's own angle change is logged; earlier entries are kept
  const s = C.newRecord(img, { blind: true });
  C.setAngle(s, 'h', 4, 'auto', { est: 4 });
  const snapRot = JSON.stringify(s);
  C.setAngle(s, 'h', 9, 'drag'); C.setAngle(s, 'v', -2, 'key');
  const u2 = C.keepAudit(JSON.parse(snapRot), s);
  assert.strictEqual(u2.h_angle_deg, 4); assert.strictEqual(C.lineAngle(u2, 'v'), 0);
  assert.deepStrictEqual(u2.angle_log.map((e) => e.how + ':' + e.line + ':' + e.deg), ['auto:h:4', 'drag:h:9', 'key:v:-2', 'undo:h:4', 'undo:v:0']);
  assert.strictEqual(C.angleSource(u2, 'h'), 'undo');
  // a v1 record (no angle_log, no angle fields) gains nothing it did not have
  const v1 = { id: 0, rect_size: S, h_line_y: 245, v_line_x: 245, h_dots: [1, 2], h_orig: [0, 0], v_dots: [], v_orig: [],
    blind_mode: false, blind_log: [], proposals_seen: true, time_spent_s: 5, n_edits: 3 };
  const cur = JSON.parse(JSON.stringify(v1)); cur.h_dots.push(3); cur.h_orig.push(0); cur.time_spent_s = 9;
  const u3 = C.keepAudit(JSON.parse(JSON.stringify(v1)), cur);
  assert.deepStrictEqual(Object.keys(u3).sort(), Object.keys(v1).sort());
  assert.strictEqual(u3.time_spent_s, 9); assert.deepStrictEqual(u3.h_dots, [1, 2]); assert.deepStrictEqual(u3.blind_log, []);
});


// ------------------------------------------------------------------ v2.2.1: turning the image (view rotation)
// a record with dots on both lines, at given line angles / positions (blind: only our own dots)
function viewRec(ah, av, hy, vx) {
  const r = C.newRecord(img, { blind: true });
  if (hy != null) C.moveLine(r, 'h', hy);
  if (vx != null) C.moveLine(r, 'v', vx);
  for (let i = 1; i < 25; i++) { C.addDot(r, 'h', i * S / 25 + 0.3); C.addDot(r, 'v', i * S / 25 - 0.2); }
  if (ah) C.setAngle(r, 'h', ah, 'key');
  if (av) C.setAngle(r, 'v', av, 'key');
  return r;
}

t('view turn is rigid: TPI, segments and dots are exactly those of turning both lines by -d', () => {
  const cases = [[0, 0, null, null], [3, -2, null, null], [0, 0, 0.04 * S, 0.97 * S], [-12.5, 7.25, 0.2 * S, 0.6 * S]];
  let full = 0;
  [0.5, -0.5, 5, -7.25, 12, -30, 33.3].forEach((d) => cases.forEach(([ah, av, hy, vx]) => {
    const a = viewRec(ah, av, hy, vx), b = clone(a);
    const screenH = C.lineAngle(a, 'h') + C.viewAngle(a), screenV = C.lineAngle(a, 'v') + C.viewAngle(a);
    const rg = C.viewRange(a), want = Math.min(rg.hi, Math.max(rg.lo, d));   // e.g. H at -12.5 can turn the image +32.5 at most
    const got = C.rotateView(a, d, 'key');
    assert.strictEqual(got, want, 'turn applied (clamped only by the 45 deg line limit)');
    if (want === d) full++;
    C.setAngle(b, 'h', ah - got, 'key'); C.setAngle(b, 'v', av - got, 'key');      // the equivalent line rotation
    assert.strictEqual(C.viewAngle(a), got);
    ['h', 'v'].forEach((l) => {
      assert.strictEqual(C.lineAngle(a, l), C.lineAngle(b, l), l + ' angle');
      assert.deepStrictEqual(C.lineInfo(a, l), C.lineInfo(b, l), l + ' lineInfo d=' + d);
      assert.deepStrictEqual(C.segment(a, l), C.segment(b, l));
      a[l + '_dots'].forEach((_, i) => assert.deepStrictEqual(C.dotXY(a, l, i), C.dotXY(b, l, i)));
    });
    assert.deepStrictEqual(C.derived(a), C.derived(b));
    assert.deepStrictEqual(C.tpis(a), C.tpis(b));
    assert.deepStrictEqual([a.h_dots, a.v_dots, a.h_line_y, a.v_line_x], [b.h_dots, b.v_dots, b.h_line_y, b.v_line_x], 'dots and pivots kept');
    // each line keeps its angle on screen (line angle + view turn)
    near(C.lineAngle(a, 'h') + C.viewAngle(a), screenH, 1e-9); near(C.lineAngle(a, 'v') + C.viewAngle(a), screenV, 1e-9);
  }));
  assert.strictEqual(full, 27, 'all but one case turn fully');
  // the count never depends on view_rot_deg itself: only the line geometry
  const r = viewRec(4, 4), q = clone(r); q.view_rot_deg = 17;
  assert.deepStrictEqual(C.derived(q), C.derived(r));
  // turning back restores the original geometry exactly
  const z = viewRec(2.5, -1), z0 = clone(z);
  C.rotateView(z, 9.37, 'key'); C.setViewAngle(z, 0, 'zero');
  assert.strictEqual(C.viewAngle(z), 0);
  ['h', 'v'].forEach((l) => assert.deepStrictEqual(C.lineInfo(z, l), C.lineInfo(z0, l)));
});

t('view turn limits: image within +-45, lines never past +-45 on the fabric; no-op does not touch', () => {
  const r = viewRec(0, 0);
  assert.strictEqual(C.rotateView(r, 50, 'key'), 45);                 // view limit
  assert.strictEqual(C.viewAngle(r), 45); assert.strictEqual(C.lineAngle(r, 'h'), -45);
  const ed = r.n_edits, ts = r.timestamp, nlog = r.angle_log.length;
  assert.strictEqual(C.rotateView(r, 0.5, 'key'), 0);
  assert.strictEqual(C.rotateView(r, 0, 'key'), 0);
  assert.strictEqual(C.rotateView(r, NaN, 'key'), 0);
  assert.deepStrictEqual([r.n_edits, r.timestamp, r.angle_log.length], [ed, ts, nlog], 'no-op leaves the record untouched');
  const q = viewRec(40, -3);                                          // H at 40: the image can turn at most -5 deg
  assert.deepStrictEqual(C.viewRange(q), { lo: -5, hi: 42 });
  assert.strictEqual(C.rotateView(q, -8, 'key'), -5);
  assert.strictEqual(C.lineAngle(q, 'h'), 45); assert.strictEqual(C.lineAngle(q, 'v'), 2); assert.strictEqual(C.viewAngle(q), -5);
  assert.strictEqual(C.rotateView(q, -0.5, 'key'), 0);
  // rounding: 0.01 deg steps, never -0
  const w = viewRec(0.33, 0);
  C.rotateView(w, 0.5, 'key');
  assert.strictEqual(C.lineAngle(w, 'h'), -0.17); assert.strictEqual(C.viewAngle(w), 0.5);
  C.rotateView(w, -0.5, 'key');
  assert.ok(Object.is(C.viewAngle(w), 0) && Object.is(w.view_rot_deg, 0) && Object.is(C.lineAngle(w, 'v'), 0));
});

t('pointer mapping: screenToWorld inverts viewToScreen at any turn / zoom / pan; turn 0 is the v2.2.0 mapping', () => {
  const pts = [[0, 0], [S, 0], [S, S], [0, S], [S / 2, S / 2], [0.137 * S, 0.81 * S], [-30, S + 12]];
  [0, 0.5, -0.5, 3, -7.25, 17, 45, -45].forEach((rot) => [0.2, 1, 3.7, 60].forEach((scale) => [[0, 0], [123.4, -56.7], [-900, 4000]].forEach(([ox, oy]) => {
    const v = { scale: scale, ox: ox, oy: oy, rot: rot };
    pts.forEach(([x, y]) => {
      const p = C.viewToScreen(v, S, x, y), w = C.screenToWorld(v, S, p[0], p[1]);
      near(w[0], x, 1e-9 * Math.max(1, Math.abs(ox) + Math.abs(oy)) / scale + 1e-9, 'x rot=' + rot); near(w[1], y, 1e-9 * Math.max(1, Math.abs(ox) + Math.abs(oy)) / scale + 1e-9, 'y');
      // and screen -> world -> screen
      const q = C.screenToWorld(v, S, x * 1.3 + 7, y * 0.7 - 3), b = C.viewToScreen(v, S, q[0], q[1]);
      near(b[0], x * 1.3 + 7, 1e-7); near(b[1], y * 0.7 - 3, 1e-7);
    });
    // a turn about the crop centre: the centre maps like the unturned view; distances scale by `scale` only
    const c0 = C.viewToScreen(v, S, S / 2, S / 2);
    near(c0[0], ox + scale * S / 2, 1e-9 * (1 + Math.abs(ox))); near(c0[1], oy + scale * S / 2, 1e-9 * (1 + Math.abs(oy)));
    const a = C.viewToScreen(v, S, 10, 20), b = C.viewToScreen(v, S, 110, 95);
    near(Math.hypot(b[0] - a[0], b[1] - a[1]), scale * 125, 1e-9 * scale * 125);
    // + = clockwise on screen (y down): the image x axis points to (cos, sin)
    const d = C.viewDir(v, [1, 0]);
    near(d[0], Math.cos(rot * Math.PI / 180), 1e-12); near(d[1], Math.sin(rot * Math.PI / 180), 1e-12);
  })));
  // turn 0: bit-identical to the v2.2.0 mapping o + scale * w (no floating-point drift)
  const v0 = { scale: 1.7, ox: 33.3, oy: -12.1, rot: 0 };
  [[0.1, 0.2], [489.9, 3.3]].forEach(([x, y]) => {
    assert.deepStrictEqual(C.viewToScreen(v0, S, x, y), [33.3 + 1.7 * x, -12.1 + 1.7 * y]);
    assert.deepStrictEqual(C.screenToWorld(v0, S, x, y), [(x - 33.3) / 1.7, (y + 12.1) / 1.7]);
  });
  assert.strictEqual(C.viewExtent(0), 1); near(C.viewExtent(45), Math.SQRT2, 1e-12); near(C.viewExtent(-10), Math.cos(Math.PI / 18) + Math.sin(Math.PI / 18), 1e-12);
});

t('clicks land where they are seen: a dot clicked on a turned image projects back onto its own parameter', () => {
  [[0, 0, 0], [3, -2, 6.5], [-12.5, 7.25, -20], [0, 0, 45]].forEach(([ah, av, rot]) => {
    const r = viewRec(ah, av, 0.3 * S, 0.55 * S);
    if (rot) C.rotateView(r, rot, 'key');
    const v = { scale: 2.3, ox: -140, oy: 75, rot: C.viewAngle(r) };
    ['h', 'v'].forEach((l) => {
      const L = C.lineInfo(r, l);
      r[l + '_dots'].forEach((p, i) => {
        if (p < L.lo || p > L.hi) return;
        const sp = C.viewToScreen(v, S, ...C.dotXY(r, l, i));       // where the dot is drawn
        const w = C.screenToWorld(v, S, sp[0], sp[1]);              // where a click there lands
        const pr = C.projectToLine(r, l, w[0], w[1]);
        near(pr.dist, 0, 1e-8, 'on the line'); near(pr.pos, p, 1e-8, 'same parameter');
      });
      // a click 5 screen px beside the line (perpendicular on screen) is 5 / scale image px from it
      const u = C.viewDir(v, C.lineDir(r, l)), mid = C.pointAt(r, l, (L.lo + L.hi) / 2);
      const sm = C.viewToScreen(v, S, mid[0], mid[1]), off = [sm[0] - 5 * u[1], sm[1] + 5 * u[0]];
      const w = C.screenToWorld(v, S, off[0], off[1]);
      near(C.projectToLine(r, l, w[0], w[1]).dist, 5 / 2.3, 1e-8);
    });
  });
});

t('box delete in screen space: exactly the dots drawn inside the box, on a turned image too', () => {
  [0, 8, -33].forEach((rot) => {
    const r = viewRec(2, -1, 0.4 * S, 0.45 * S);
    if (rot) C.rotateView(r, rot, 'key');
    const v = { scale: 1.6, ox: 30, oy: -10, rot: C.viewAngle(r) };
    const box = [0.2 * S, 0.1 * S, 1.1 * S, 0.9 * S];
    const inBox = (l, i) => { const p = C.viewToScreen(v, S, ...C.dotXY(r, l, i)); return p[0] >= box[0] && p[0] <= box[2] && p[1] >= box[1] && p[1] <= box[3]; };
    const keepH = r.h_dots.filter((_, i) => !inBox('h', i)), keepV = r.v_dots.filter((_, i) => !inBox('v', i));
    const n0 = r.h_dots.length + r.v_dots.length, e0 = r.n_edits;
    const n = C.boxDeleteScreen(r, v, box[2], box[3], box[0], box[1]);     // corners in any order
    assert.ok(n > 0 && n < n0);
    assert.strictEqual(n, n0 - keepH.length - keepV.length);
    assert.deepStrictEqual(r.h_dots, keepH); assert.deepStrictEqual(r.v_dots, keepV);
    assert.strictEqual(r.h_orig.length, r.h_dots.length); assert.strictEqual(r.n_edits, e0 + 1);
    assert.strictEqual(C.boxDeleteScreen(r, v, -50, -50, -40, -40), 0);
    assert.strictEqual(r.n_edits, e0 + 1, 'empty box: no edit');
  });
  // turn 0: same as the image-space box delete
  const a = viewRec(0, 0, 100, 200), b = clone(a), v = { scale: 2, ox: 10, oy: 20, rot: 0 };
  assert.strictEqual(C.boxDeleteScreen(a, v, 10 + 2 * 120, 20 + 2 * 80, 10 + 2 * 260, 20 + 2 * 160), C.boxDelete(b, 120, 80, 260, 160));
  assert.deepStrictEqual([a.h_dots, a.v_dots], [b.h_dots, b.v_dots]);
});

t('moving a line on a turned image keeps the grabbed point under the pointer (pivotThrough)', () => {
  [[0, 0], [7, -4], [-30, 22]].forEach(([ah, av]) => {
    const r = viewRec(ah, av, 0.3 * S, 0.6 * S);
    ['h', 'v'].forEach((l) => {
      const grab = C.pointAt(r, l, 0.35 * S), pv = C.linePivot(r, l);
      const off = (l === 'h' ? pv[1] : pv[0]) - C.pivotThrough(r, l, grab[0], grab[1]);
      near(off, 0, 1e-9, 'a point on the line needs no offset');
      const to = [grab[0] + 13.7, grab[1] - 21.1];                         // drag anywhere
      C.moveLine(r, l, C.pivotThrough(r, l, to[0], to[1]) + off);
      near(C.projectToLine(r, l, to[0], to[1]).dist, 0, 0.06, 'pointer on the line (0.1 px pivot rounding)');
    });
  });
  const lv = viewRec(0, 0);
  assert.strictEqual(C.pivotThrough(lv, 'h', 12, 34), 34); assert.strictEqual(C.pivotThrough(lv, 'v', 12, 34), 12);
});

t('view turns are logged (how=view, per line, bursts coalesced, align carries the estimate); undo and reset keep the trail', () => {
  const r = viewRec(0, 0), n0 = r.angle_log.length;
  C.rotateView(r, 0.5, 'key'); C.rotateView(r, 0.5, 'key'); C.rotateView(r, 5, 'key');
  const vlog = r.angle_log.slice(n0);
  assert.strictEqual(vlog.length, 2, 'one pair for a key burst');
  assert.deepStrictEqual(vlog.map((e) => [e.line, e.deg, e.how, e.view, e.src]), [['h', -6, 'view', 6, 'key'], ['v', -6, 'view', 6, 'key']]);
  assert.strictEqual(C.angleSource(r, 'h'), 'view');
  C.setViewAngle(r, 2, 'slider');
  C.rotateView(r, C.viewAlignDelta(r, 'h', 1.5), 'align', { est: 1.5, clear: true, ref: 'h' });
  const last = r.angle_log.slice(-2);
  assert.deepStrictEqual(last.map((e) => [e.line, e.deg, e.how, e.src, e.est, e.ref]), [['h', 1.5, 'view', 'align', 1.5, 'h'], ['v', 1.5, 'view', 'align', 1.5, 'h']]);
  assert.strictEqual(C.viewAngle(r), -1.5, 'level line on screen -> image turned by -estimate');
  near(C.lineAngle(r, 'h') + C.viewAngle(r), 0, 1e-12, 'the line is still level on screen');
  assert.strictEqual(C.viewAlignDelta(r, 'h', 1.5), 0, 'aligned now');
  // undo of a view turn: geometry restored, trail append-only, the undo's turn logged with the view
  const snap = JSON.stringify(r), nl = r.angle_log.length;
  C.rotateView(r, -3, 'key');
  const u = C.keepAudit(JSON.parse(snap), r);
  assert.strictEqual(C.viewAngle(u), -1.5); assert.strictEqual(C.lineAngle(u, 'h'), 1.5);
  assert.deepStrictEqual(u.angle_log.slice(0, nl + 2), r.angle_log.slice(0, nl + 2));
  assert.deepStrictEqual(u.angle_log.slice(-2).map((e) => [e.line, e.deg, e.how, e.view]), [['h', 1.5, 'undo', -1.5], ['v', 1.5, 'undo', -1.5]]);
  // undo to a snapshot from before the first turn (no view field): back to 0, logged
  const fresh = viewRec(0, 0), s0 = JSON.stringify(fresh);
  C.rotateView(fresh, 4, 'key');
  const u0 = C.keepAudit(JSON.parse(s0), fresh);
  assert.strictEqual(C.viewAngle(u0), 0); assert.ok(!('view_rot_deg' in u0));
  assert.deepStrictEqual(u0.angle_log.slice(-2).map((e) => [e.how, e.deg, e.view]), [['undo', 0, 0], ['undo', 0, 0]]);
  // reset to proposals turns the image back too (one reset entry per line, with view 0)
  const p = C.newRecord(img, {});
  C.rotateView(p, -6, 'key'); C.setAngle(p, 'h', 0, 'zero');
  C.resetToProposals(p, img);
  assert.strictEqual(C.viewAngle(p), 0); assert.strictEqual(C.lineAngle(p, 'h'), 0); assert.strictEqual(C.lineAngle(p, 'v'), 0);
  assert.deepStrictEqual(p.angle_log.slice(-2).map((e) => [e.line, e.how, e.deg, e.view]), [['h', 'reset', 0, 0], ['v', 'reset', 0, 0]]);
  assert.strictEqual(p.h_dots.length, P.h_positions.length);
});

t('records with and without view_rot_deg: JSON and CSV round trips; CSV column appended at the end only', () => {
  // without: a new record never gets the field; v1 / v2.2.0 records load and export unchanged
  const plain = C.newRecord(Object.assign({}, img, { id: 101 }), { annotator: 'eve' });   // ids apart from the v1 record (id 0)
  assert.ok(!('view_rot_deg' in plain));
  assert.strictEqual(C.viewAngle(plain), 0); assert.strictEqual(C.viewAngle(clone(V1_REC)), 0);
  const turned = C.newRecord(Object.assign({}, img, { id: 102 }), { annotator: 'eve' });
  C.rotateView(turned, -3.75, 'key'); C.setAngle(turned, 'v', 2, 'key');
  const store = C.emptyStore('eve');
  store.records[String(plain.id)] = plain; store.records[String(turned.id)] = turned; store.records['0'] = Object.assign(clone(V1_REC), { annotator: 'eve' });
  const before = clone(store.records);
  const ex = JSON.parse(JSON.stringify(C.exportJSON(store, manifest)));
  assert.deepStrictEqual(store.records, before, 'export never mutates the store');
  const ej = (id) => ex.records.find((r) => r.id === id);
  assert.ok(!('view_rot_deg' in ej(plain.id)) && !('view_rot_deg' in ej(0)), 'no field for unturned records');
  assert.strictEqual(ej(turned.id).view_rot_deg, -3.75);
  const back = C.parseImport(ex).records;
  assert.deepStrictEqual(back[String(turned.id)], C.withDerived(turned));
  assert.deepStrictEqual(back[String(plain.id)], C.withDerived(plain));
  Object.keys(V1_REC).forEach((k) => k === 'annotator' || assert.deepStrictEqual(back['0'][k], V1_REC[k], k));
  // CSV: 47 v1 + 15 v2 columns unchanged and first, view_rot_deg last
  assert.strictEqual(C.CSV_COLS.length, 63);
  assert.deepStrictEqual(C.CSV_COLS, C.CSV_COLS_V1.concat(C.CSV_COLS_V2, ['view_rot_deg']));
  assert.strictEqual(C.CSV_COLS.indexOf('view_rot_deg'), C.CSV_COLS.length - 1);
  assert.strictEqual(C.CSV_COLS.slice(0, 47).join(','), V1_CSV_HEADER);
  const csv = C.exportCSV(store), rows = C.parseCSV(csv), H = rows[0];
  assert.deepStrictEqual(H, C.CSV_COLS);
  rows.forEach((row) => assert.strictEqual(row.length, 63));
  const row = (id) => Object.fromEntries(H.map((h, i) => [h, rows.find((x) => x[0] === String(id))[i]]));
  assert.strictEqual(row(turned.id).view_rot_deg, '-3.75'); assert.strictEqual(row(plain.id).view_rot_deg, '0'); assert.strictEqual(row(0).view_rot_deg, '0');
  assert.strictEqual(row(turned.id).h_angle_deg, '3.75'); assert.strictEqual(row(turned.id).v_angle_deg, '2');
  assert.strictEqual(row(turned.id).h_angle_src, 'view'); assert.strictEqual(row(turned.id).v_angle_src, 'key');
  assert.strictEqual(csv.split('\n')[0], C.CSV_COLS.join(','));
  const csvBack = C.parseImportCSV(csv).records;
  assert.strictEqual(csvBack[String(turned.id)].view_rot_deg, -3.75);
  ['h', 'v'].forEach((l) => assert.deepStrictEqual(C.lineInfo(csvBack[String(turned.id)], l), C.lineInfo(turned, l)));
  assert.ok(!('view_rot_deg' in csvBack[String(plain.id)]) && !('view_rot_deg' in csvBack['0']), 'no field when the column is 0');
  // a v2.2.0 CSV (62 columns, no view_rot_deg) still imports; so does a v1 CSV
  const old = rows.map((r) => r.slice(0, 62).map((c) => (/[",\n]/.test(c) ? '"' + c.replace(/"/g, '""') + '"' : c)).join(',')).join('\n');
  const oldBack = C.parseImportCSV(old).records;
  assert.ok(!('view_rot_deg' in oldBack[String(turned.id)]));
  assert.strictEqual(oldBack[String(turned.id)].h_angle_deg, 3.75);
  assert.deepStrictEqual(C.lineInfo(oldBack[String(turned.id)], 'h'), C.lineInfo(turned, 'h'), 'the count needs no view field');
});

if (manifest) {
  t('manifest sanity: 150 images, files exist, proposals inside the crop', () => {
    assert.strictEqual(manifest.images.length, 150);
    assert.strictEqual(manifest.iaa_subset.length, 20);
    assert.strictEqual(new Set(manifest.iaa_subset).size, 20);
    manifest.iaa_subset.forEach((i) => assert.ok(manifest.images.some((e) => e.id === i)));
    manifest.images.forEach((e) => {
      ['crop', 'crop_enh', 'thumb'].forEach((k) => assert.ok(fs.existsSync(path.join(__dirname, '..', e[k])), e[k]));
      const s = e.rect_size[0], p = e.proposal;
      assert.ok(p.h_line_y >= 0 && p.h_line_y <= s && p.v_line_x >= 0 && p.v_line_x <= s);
      p.h_positions.concat(p.v_positions).forEach((x) => assert.ok(x >= 0 && x <= s));
    });
  });
}

function parseCSV(text) {
  const rows = []; let row = [], cell = '', q = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (q) {
      if (ch === '"' && text[i + 1] === '"') { cell += '"'; i++; } else if (ch === '"') q = false; else cell += ch;
    } else if (ch === '"') q = true;
    else if (ch === ',') { row.push(cell); cell = ''; }
    else if (ch === '\n') { row.push(cell); rows.push(row); row = []; cell = ''; }
    else cell += ch;
  }
  if (cell || row.length) { row.push(cell); rows.push(row); }
  return rows;
}

console.log(`\nall ${n} tests passed`);
