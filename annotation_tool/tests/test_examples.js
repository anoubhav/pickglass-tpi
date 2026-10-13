// Headless tests of the v2.2.2 Examples panel logic (core.js): the "My finished scenes" gallery filter (own
// records only, done only, IAA handling), the example-figure filter (never an IAA-subset scene, hidden when the
// scene is counted blind) and the thumbnail geometry (image turn + line angles).
//   node tests/test_examples.js
'use strict';
const assert = require('assert');
const fs = require('fs');
const path = require('path');
const C = require('../core.js');

const ROOT = path.join(__dirname, '..');
const manifest = JSON.parse(fs.readFileSync(path.join(ROOT, 'data', 'manifest.json'), 'utf8'));
const images = manifest.images.slice().sort((a, b) => a.id - b.id);
const byId = {}; images.forEach((e) => { byId[e.id] = e; });
const iaa = new Set(manifest.iaa_subset);          // authoritative list: the manifest
const nReps = images.filter((e) => !C.isCopy(e)).length;
let n = 0;
const t = (name, fn) => { fn(); n++; console.log('ok', n, name); };
const near = (a, b, tol, msg) => assert.ok(Math.abs(a - b) <= (tol || 1e-6), (msg || '') + ' ' + a + ' vs ' + b);

const nonIaaRep = images.find((e) => !C.isCopy(e) && !iaa.has(e.id)).id;
const iaaRep = manifest.iaa_subset[0];
const copyOfIaa = images.find((e) => C.isCopy(e) && iaa.has(C.sceneRepId(e)));
const copyOfNonIaa = images.find((e) => C.isCopy(e) && !iaa.has(C.sceneRepId(e)));
let clock = Date.parse('2026-10-08T12:00:00Z');

/** A record of `id` by `who` with n dots per line; done unless stated; timestamps increase with each call. */
function rec(id, who, extra, nh, nv) {
  const img = byId[id], inIaa = iaa.has(id) || iaa.has(C.sceneRepId(img));
  const r = C.newRecord(img, { annotator: who, blind: inIaa, iaa: iaa.has(id) });
  if (!inIaa) C.removeProposalDots(r);
  const S = r.rect_size;
  nh = nh == null ? 40 : nh; nv = nv == null ? 30 : nv;
  for (let i = 0; i < nh; i++) C.addDot(r, 'h', (i + 0.5) * S / nh);
  for (let i = 0; i < nv; i++) C.addDot(r, 'v', (i + 0.5) * S / nv);
  r.weave = 'plain'; r.confidence = 3; r.done = true;
  clock += 60000; r.timestamp = new Date(clock).toISOString();
  return Object.assign(r, extra || {});
}
const store = (...rs) => { const m = {}; rs.forEach((r) => { m[String(r.id)] = r; }); return m; };

t('test fixtures: the manifest has the IAA v2 subset, a non-IAA rep, an IAA rep and copies of both', () => {
  assert.strictEqual(iaa.size, 20);
  assert.ok(!iaa.has(nonIaaRep) && iaa.has(iaaRep) && copyOfIaa && copyOfNonIaa);
  assert.strictEqual(nReps, 76);
});

// ------------------------------------------------------------------ gallery: own records only
t('gallery: only records whose annotator is the current name (never another annotator, never unnamed ones)', () => {
  const ids = images.filter((e) => !C.isCopy(e) && !iaa.has(e.id)).slice(0, 6).map((e) => e.id);
  const records = store(
    rec(ids[0], 'alice'), rec(ids[1], 'alice'),
    rec(ids[2], 'bob'),                 // another annotator's record in the same records map (e.g. an import)
    rec(ids[3], ''),                    // no annotator
    rec(ids[4], 'anonymous'),           // the safeName of '' -- still not "alice"
    rec(ids[5], 'Alice'));              // different storage key (safeName is case-sensitive) -> not alice's
  const g = C.finishedScenes(records, 'alice', images, iaa);
  assert.strictEqual(g.needName, false);
  assert.deepStrictEqual(g.items.map((x) => x.id).sort((a, b) => a - b), [ids[0], ids[1]].sort((a, b) => a - b));
  const gb = C.finishedScenes(records, 'bob', images, iaa);
  assert.deepStrictEqual(gb.items.map((x) => x.id), [ids[2]]);
  // a name is compared as stored: surrounding blanks do not matter, safeName-equal spellings share a store
  assert.deepStrictEqual(C.finishedScenes(records, '  alice ', images, iaa).items.length, 2);
  assert.strictEqual(C.sameAnnotator('Bob Smith', 'Bob_Smith'), true);
  assert.strictEqual(C.sameAnnotator('', ''), false);
  assert.strictEqual(C.sameAnnotator('anonymous', ''), false);
});

t('gallery: an empty Annotator name shows a prompt, never a gallery (not even of unnamed records)', () => {
  const records = store(rec(nonIaaRep, ''), rec(iaaRep, ''));
  [undefined, null, '', '   '].forEach((name) => {
    const g = C.finishedScenes(records, name, images, iaa);
    assert.strictEqual(g.needName, true);
    assert.deepStrictEqual(g.items, []);
    assert.strictEqual(g.nDone, 0);
    assert.strictEqual(g.total, nReps);
  });
});

t('gallery: two annotators doing the same blind subset never see each other\'s scenes', () => {
  // the same 20 IAA scenes done by A, then by B (separate stores, as the app keeps them, and one mixed map)
  const A = store(...manifest.iaa_subset.map((id) => rec(id, 'annA', {}, 41, 33)));
  const B = store(...manifest.iaa_subset.slice(0, 5).map((id) => rec(id, 'annB', {}, 44, 35)));
  const gB = C.finishedScenes(B, 'annB', images, iaa);
  assert.strictEqual(gB.items.length, 5);
  assert.ok(gB.items.every((x) => x.iaa && x.tpi === undefined));
  assert.strictEqual(C.finishedScenes(A, 'annB', images, iaa).items.length, 0);   // B's name on A's store: nothing
  const mixed = Object.assign({}, A);
  Object.keys(B).forEach((k) => { mixed[k + '_b'] = B[k]; });
  assert.strictEqual(C.finishedScenes(mixed, 'annB', images, iaa).items.length, 5);
  assert.strictEqual(C.finishedScenes(mixed, 'annA', images, iaa).items.length, 20);
});

// ------------------------------------------------------------------ gallery: done only, order, counts
t('gallery: done records only, most recent first; n done counts scene reps (of 76), copies are listed but not counted', () => {
  const ids = images.filter((e) => !C.isCopy(e) && !iaa.has(e.id)).slice(0, 4).map((e) => e.id);
  const notDone = rec(ids[0], 'alice', { done: false });
  const pristine = C.newRecord(byId[ids[1]], { annotator: 'alice' });   // opened, never edited
  const old = rec(ids[2], 'alice');
  const copy = rec(copyOfNonIaa.id, 'alice');                            // an unlocked copy, done
  const newest = rec(ids[3], 'alice');
  const g = C.finishedScenes(store(notDone, pristine, old, copy, newest), 'alice', images, iaa);
  assert.deepStrictEqual(g.items.map((x) => x.id), [newest.id, copy.id, old.id]);
  assert.strictEqual(g.nDone, 2);
  assert.strictEqual(g.total, 76);
  const c = g.items.find((x) => x.id === copy.id);
  assert.strictEqual(c.copy, true);
  assert.strictEqual(c.rep, C.sceneRepId(copyOfNonIaa));
  // equal timestamps: by id
  const a = rec(ids[0], 'alice'), b = rec(ids[1], 'alice'); b.timestamp = a.timestamp;
  assert.deepStrictEqual(C.finishedScenes(store(b, a), 'alice', images, iaa).items.map((x) => x.id), [ids[0], ids[1]].sort((p, q) => p - q));
});

t('gallery: per item threads/inch (incl. turned lines / image), weave, confidence, flags and empty (cleared) lines', () => {
  const r = rec(nonIaaRep, 'alice', { weave: 'twill_2_1', confidence: 2, nonwoven: true }, 46, 39);
  r.flags.glare = true; r.flags.blur = true;
  C.rotateView(r, -5, 'key'); C.setAngle(r, 'h', 8, 'key');
  const it = C.finishedScenes(store(r), 'alice', images, iaa).items[0];
  near(it.tpi.h, C.lineInfo(r, 'h').tpi); near(it.tpi.v, C.lineInfo(r, 'v').tpi);
  assert.ok(Math.abs(it.tpi.h - 46) > 0.1, 'a turned line is normalised');
  assert.deepStrictEqual(it.angle, { h: 8, v: 5 });
  assert.strictEqual(it.view, -5);
  assert.strictEqual(it.weave, 'twill_2_1'); assert.strictEqual(it.confidence, 2);
  assert.deepStrictEqual(it.flags, ['glare', 'blur']); assert.strictEqual(it.nonwoven, true);
  assert.deepStrictEqual(it.empty, []);
  const cl = rec(nonIaaRep, 'alice'); C.clearLine(cl, 'v');
  assert.deepStrictEqual(C.finishedScenes(store(cl), 'alice', images, iaa).items[0].empty, ['v']);
  const both = rec(nonIaaRep, 'alice', {}, 0, 0);
  assert.deepStrictEqual(C.finishedScenes(store(both), 'alice', images, iaa).items[0].empty, ['h', 'v']);
});

// ------------------------------------------------------------------ gallery: IAA handling
t('gallery IAA: IAA-subset scenes are listed (own, done) without counts, angles or turn; copies of IAA scenes too', () => {
  const a = rec(iaaRep, 'alice', {}, 40, 30);
  const c = rec(copyOfIaa.id, 'alice', {}, 40, 30);    // an unlocked copy of an IAA scene
  const g = C.finishedScenes(store(a, c), 'alice', images, iaa);
  assert.strictEqual(g.items.length, 2);
  g.items.forEach((it) => {
    assert.strictEqual(it.iaa, true);
    // nothing count-like: no tpi / angle / view, only these keys
    assert.deepStrictEqual(Object.keys(it).sort(),
      ['confidence', 'copy', 'empty', 'flags', 'iaa', 'id', 'nonwoven', 'rep', 't', 'timestamp', 'weave']);
  });
  assert.strictEqual(g.nDone, 1);   // the rep counts as a done scene, the copy does not
  // an empty line on an IAA scene is still flagged (a likely mistake), without any count
  const e = rec(iaaRep, 'alice'); C.clearLine(e, 'h');
  assert.deepStrictEqual(C.finishedScenes(store(e), 'alice', images, iaa).items[0].empty, ['h']);
});

t('gallery IAA: membership comes from the manifest list; a record\'s own iaa flag only ever adds (never removes) IAA', () => {
  const stale = rec(nonIaaRep, 'alice', { iaa: true });     // e.g. opened under the retired v1 subset
  assert.strictEqual(C.finishedScenes(store(stale), 'alice', images, iaa).items[0].iaa, true);
  const unflagged = rec(iaaRep, 'alice', { iaa: false });
  assert.strictEqual(C.finishedScenes(store(unflagged), 'alice', images, iaa).items[0].iaa, true);
  // with an empty subset nothing is IAA (except by the record's own flag)
  assert.strictEqual(C.finishedScenes(store(rec(iaaRep, 'alice', { iaa: false })), 'alice', images, new Set()).items[0].iaa, false);
});

t('gallery: read-only (records are not changed) and unknown ids / non-records are skipped', () => {
  const records = store(rec(nonIaaRep, 'alice'), rec(iaaRep, 'alice'));
  records['9999'] = Object.assign(rec(nonIaaRep, 'alice'), { id: 9999 });
  records.junk = { id: 5, done: true, annotator: 'alice' };   // no dots arrays: not a record
  const before = JSON.stringify(records);
  const g = C.finishedScenes(records, 'alice', images, iaa);
  assert.strictEqual(JSON.stringify(records), before);
  assert.deepStrictEqual(g.items.map((x) => x.id).sort((a, b) => a - b), [nonIaaRep, iaaRep].sort((a, b) => a - b));
});

// ------------------------------------------------------------------ example figures
const index = JSON.parse(fs.readFileSync(path.join(ROOT, 'guide', 'examples', 'examples.json'), 'utf8'));

t('figures (examples.json): every figure names its scenes; none is an IAA-subset scene, a copy or unknown; files exist', () => {
  assert.ok(index.figures.length >= 4);
  const v = C.visibleExamples(index, images, {}, iaa, { policy: 'iaa' });
  assert.strictEqual(v.refused, 0);
  assert.strictEqual(v.blind, 0);
  assert.strictEqual(v.shown.length, index.figures.length);
  index.figures.forEach((f) => {
    assert.ok(Array.isArray(f.scenes) && f.scenes.length, f.src);
    f.scenes.forEach((id) => {
      assert.ok(byId[id], 'unknown scene ' + id);
      assert.ok(!C.isCopy(byId[id]), 'copy ' + id);
      assert.ok(!iaa.has(id) && !iaa.has(C.sceneRepId(byId[id])), 'IAA scene in ' + f.src + ': ' + id);
    });
    assert.ok(fs.existsSync(path.join(ROOT, f.src)), f.src);
    assert.notStrictEqual(f.src, 'guide/annotation_guide.png');   // shows proposal dots on scene 77 (IAA)
  });
  assert.deepStrictEqual(index.iaa_subset_excluded, manifest.iaa_subset.slice().sort((a, b) => a - b));
  // ~4 per-scene proposal figures covering full and quarter resolution
  const per = index.figures.filter((f) => /example_\d+\.jpg$/.test(f.src));
  assert.ok(per.length >= 4);
  assert.ok(per.some((f) => f.resolution === 'full') && per.some((f) => f.resolution === 'quarter'));
  per.forEach((f) => assert.strictEqual(f.caption, 'machine proposals — shows the layout, not verified counts'));
});

t('figures: an IAA id (or a copy of an IAA scene) is refused even if listed; so are unknown ids, missing scenes, odd paths', () => {
  const fig = (scenes, src) => ({ src: src || 'guide/examples/x.jpg', scenes: scenes, title: 't', caption: 'c' });
  const bad = { figures: [
    fig([iaaRep]), fig([nonIaaRep, iaaRep]), fig([copyOfIaa.id]), fig([copyOfNonIaa.id]), fig([9999]), fig([]),
    { src: 'guide/examples/x.jpg' }, fig([nonIaaRep], 'guide/annotation_guide.png'.replace('guide/', 'http://x/')),
    fig([nonIaaRep], 'guide/../data/saved/a.png'), fig([nonIaaRep], 'javascript:alert(1)'),
  ] };
  const v = C.visibleExamples(bad, images, {}, iaa, {});
  assert.strictEqual(v.shown.length, 0);
  assert.strictEqual(v.refused, bad.figures.length);
  // every one of the 20 IAA scenes is refused
  manifest.iaa_subset.forEach((id) => assert.strictEqual(C.visibleExamples({ figures: [fig([id])] }, images, {}, iaa, {}).shown.length, 0, String(id)));
  // the same figure with a non-IAA rep is shown
  assert.strictEqual(C.visibleExamples({ figures: [fig([nonIaaRep])] }, images, {}, iaa, {}).shown.length, 1);
  assert.strictEqual(C.visibleExamples(null, images, {}, iaa, {}).shown.length, 0);
});

t('figures: hidden while the annotator counts the scene blind (record blind, or it will open blind under the policy)', () => {
  const f = { figures: [{ src: 'guide/examples/example_1.jpg', scenes: [nonIaaRep], title: 't', caption: 'c' }] };
  const blindRec = C.newRecord(byId[nonIaaRep], { annotator: 'a', blind: true });
  assert.deepStrictEqual(C.visibleExamples(f, images, store(blindRec), iaa, { policy: 'iaa' }), { shown: [], blind: 1, refused: 0 });
  const seen = C.newRecord(byId[nonIaaRep], { annotator: 'a', blind: false });
  assert.strictEqual(C.visibleExamples(f, images, store(seen), iaa, { policy: 'all' }).shown.length, 1);   // already open with proposals
  assert.strictEqual(C.visibleExamples(f, images, {}, iaa, { policy: 'all' }).blind, 1);
  assert.strictEqual(C.visibleExamples(f, images, {}, iaa, { policy: 'none' }).shown.length, 1);
  assert.strictEqual(C.visibleExamples(f, images, {}, iaa, { policy: 'manual', manualDefault: true }).blind, 1);
  assert.strictEqual(C.visibleExamples(f, images, {}, iaa, { policy: 'random' }).blind, C.hashBlind(nonIaaRep) ? 1 : 0);
  // the real index under 'all' with no records: everything hidden
  assert.strictEqual(C.visibleExamples(index, images, {}, iaa, { policy: 'all' }).shown.length, 0);
});

// ------------------------------------------------------------------ thumbnail geometry
t('thumbnail: turned image + turned lines map exactly like the main view (dots on the lines, screen angle = line + image turn)', () => {
  const r = rec(nonIaaRep, 'alice', {}, 46, 39);
  C.rotateView(r, -5, 'key'); C.setAngle(r, 'h', 8, 'key'); C.setAngle(r, 'v', 3, 'key');
  const T = 200, g = C.thumbGeometry(r, T, 4), S = r.rect_size;
  assert.strictEqual(g.view.rot, -5);
  const scr = (A, B) => Math.atan2(B[1] - A[1], B[0] - A[0]) * 180 / Math.PI;
  near(scr(g.lines.h[0], g.lines.h[1]), 8 - 5, 1e-6, 'H on screen');
  near(scr(g.lines.v[0], g.lines.v[1]) - 90, 3 - 5, 1e-6, 'V on screen');
  ['h', 'v'].forEach((l) => {
    assert.strictEqual(g.dots[l].length, r[l + '_dots'].length);
    const [A, B] = g.lines[l], L = Math.hypot(B[0] - A[0], B[1] - A[1]);
    g.dots[l].forEach((p, i) => {   // on the segment (collinear) and equal to viewToScreen of the record's dot
      near(((B[0] - A[0]) * (p[1] - A[1]) - (B[1] - A[1]) * (p[0] - A[0])) / L, 0, 1e-6, 'collinear');
      const w = C.dotXY(r, l, i), q = C.viewToScreen(g.view, S, w[0], w[1]);
      near(p[0], q[0]); near(p[1], q[1]);
    });
  });
  // the turned window fits inside the thumbnail with the padding, centred
  g.window.forEach((p) => { assert.ok(p[0] >= 4 - 1e-6 && p[0] <= T - 4 + 1e-6 && p[1] >= 4 - 1e-6 && p[1] <= T - 4 + 1e-6); });
  const cx = g.window.reduce((s, p) => s + p[0], 0) / 4, cy = g.window.reduce((s, p) => s + p[1], 0) / 4;
  near(cx, T / 2, 1e-6); near(cy, T / 2, 1e-6);
  // no turn: the plain fit, level lines
  const z = rec(nonIaaRep, 'alice'), gz = C.thumbGeometry(z, T, 0);
  near(gz.window[0][0], 0); near(gz.window[2][0], T); near(gz.lines.h[0][1], gz.lines.h[1][1]);
});

// ------------------------------------------------------------------ taking over unnamed work (app.js)
t('app.js: a newly typed name takes over unnamed annotations only after the person confirms they are theirs', () => {
  const src = fs.readFileSync(path.join(ROOT, 'app.js'), 'utf8');
  const fn = src.slice(src.indexOf('async function switchAnnotatorNow'), src.indexOf('function setHelp'));
  assert.ok(fn.length > 100 && /if \(adopt && !confirmAdopt\(edited, name\)\) \{ adopt = false;/.test(fn), 'adoption must be confirmed');
  assert.ok(fn.indexOf('confirmAdopt(') < fn.indexOf('r.annotator = name'), 'the question comes before any record is renamed');
  assert.ok(/if \(adopt\) \{/.test(fn) && !/edited\.length && !Object\.keys\(next\.records\)\.length\) \{/.test(fn), 'no unconditional adoption');
  const ca = src.slice(src.indexOf('function confirmAdopt'), src.indexOf('async function switchAnnotatorNow'));
  assert.ok(ca.length > 50 && !/_dots|tpi|weave|\.id\b/.test(ca), 'the question shows no dots, counts, weave or image ids');
});

console.log('\nall ' + n + ' example-panel tests passed');
