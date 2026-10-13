// Headless tests of the v2.1 scene logic (core.js) and of the scene fields in data/manifest.json:
//   node tests/test_scenes.js
'use strict';
const assert = require('assert');
const fs = require('fs');
const path = require('path');
const C = require('../core.js');

const ROOT = path.join(__dirname, '..');
const manifest = JSON.parse(fs.readFileSync(path.join(ROOT, 'data', 'manifest.json'), 'utf8'));
const images = manifest.images.slice().sort((a, b) => a.id - b.id);
const byId = {}; images.forEach((e) => { byId[e.id] = e; });
const iaa = new Set(manifest.iaa_subset);
let n = 0;
const t = (name, fn) => { fn(); n++; console.log('ok', n, name); };

// a record with level lines and given dot counts (TPI == count)
function rec(img, nh, nv, extra) {
  const r = C.newRecord(img, { blind: true });
  const S = r.rect_size;
  for (let i = 0; i < nh; i++) C.addDot(r, 'h', (i + 0.5) * S / nh);
  for (let i = 0; i < nv; i++) C.addDot(r, 'v', (i + 0.5) * S / nv);
  return Object.assign(r, extra || {});
}

t('manifest: scene fields on all 150 images; 76 scenes with exactly one rep each', () => {
  assert.ok(C.hasScenes(manifest));
  assert.strictEqual(images.length, 150);
  const reps = images.filter((e) => e.is_scene_rep);
  assert.strictEqual(reps.length, 76);
  assert.strictEqual(new Set(images.map((e) => e.scene_id)).size, 76);
  reps.forEach((r) => assert.strictEqual(r.scene_id, r.id));
  images.forEach((e) => {
    const rep = byId[e.scene_id];
    assert.ok(rep && rep.is_scene_rep, 'scene_id must name a rep: ' + e.id);
    assert.deepStrictEqual(e.scene_members, rep.scene_members);
    assert.ok(e.scene_members.includes(e.id));
    assert.ok([0, 1, 2, 3].includes(e.scene_rot_k));
    assert.strictEqual(e.scene_swap_hv, e.scene_rot_k % 2 === 1);
    assert.strictEqual(e.kind === 'rep', e.is_scene_rep);
    if (e.is_scene_rep) assert.strictEqual(e.scene_rot_k, 0);
  });
  assert.deepStrictEqual(manifest.scenes.reps, reps.map((r) => r.id));
});

t('manifest: kinds 76 rep / 6 byte_identical / 47 quarter_res_copy / 10 png_copy / 11 notebook_synthetic', () => {
  const c = {};
  images.forEach((e) => { c[e.kind] = (c[e.kind] || 0) + 1; });
  assert.deepStrictEqual(c, { rep: 76, byte_identical: 6, quarter_res_copy: 47, png_copy: 10, notebook_synthetic: 11 });
  const syn = images.filter((e) => e.kind === 'notebook_synthetic').map((e) => e.id);
  assert.deepStrictEqual(syn, [131, 132, 134, 135, 136, 137, 138, 139, 141, 142, 144]);
  const png = images.filter((e) => e.kind === 'png_copy').map((e) => e.id);
  assert.deepStrictEqual(png, [129, 130, 133, 140, 143, 145, 146, 147, 148, 149]);
  // a notebook file is turned +-90 deg relative to the file it was made from
  images.filter((e) => e.kind === 'notebook_synthetic').forEach((e) => {
    assert.strictEqual(((e.scene_rot_k - byId[e.copy_of].scene_rot_k) % 4 + 4) % 4 % 2, 1, 'synthetic ' + e.id);
  });
  // exact copies keep the orientation of the file they copy
  images.filter((e) => e.kind === 'byte_identical' || e.kind === 'png_copy').forEach((e) => {
    assert.strictEqual(e.scene_rot_k, byId[e.copy_of].scene_rot_k, 'copy ' + e.id);
  });
});

t('manifest: scenes_to_annotate.txt lists exactly the reps and their members', () => {
  const txt = fs.readFileSync(path.join(ROOT, 'scenes_to_annotate.txt'), 'utf8');
  const rows = txt.split('\n').filter((l) => /^\d+:/.test(l)).map((l) => {
    const [k, v] = l.split(':');
    return [Number(k), JSON.parse(v.trim())];
  });
  assert.strictEqual(rows.length, 76);
  rows.forEach(([rep, members]) => {
    assert.ok(byId[rep].is_scene_rep);
    assert.deepStrictEqual(members, byId[rep].scene_members);
  });
});

t('manifest: IAA v2 = 20 reps of 20 distinct scenes; v1 subset kept in iaa_history', () => {
  assert.strictEqual(manifest.iaa_version, 2);
  assert.strictEqual(manifest.iaa_subset.length, 20);
  manifest.iaa_subset.forEach((i) => assert.ok(byId[i].is_scene_rep, 'IAA id ' + i + ' is not a rep'));
  assert.strictEqual(new Set(manifest.iaa_subset.map((i) => byId[i].scene_id)).size, 20);
  const h = manifest.iaa_history;
  assert.ok(Array.isArray(h) && h.length >= 1);
  assert.deepStrictEqual(h[0].subset, [1, 4, 10, 17, 19, 24, 33, 36, 41, 43, 48, 53, 61, 68, 69, 74, 86, 104, 115, 148]);
  assert.strictEqual(h[0].version, 1);
  const st = manifest.iaa_policy.strata;
  assert.strictEqual(st.reduce((s, x) => s + x.n_drawn, 0), 20);
  const full = manifest.iaa_subset.filter((i) => Math.max(byId[i].orig_w, byId[i].orig_h) >= 3000).length;
  assert.strictEqual(full, 13);   // 50 of 76 scenes are full resolution -> 13.2 of 20
});

t('isCopy / sceneRepId / sceneRotK', () => {
  assert.ok(!C.isCopy(byId[9]) && C.isCopy(byId[0]));
  assert.strictEqual(C.sceneRepId(byId[0]), 9);
  assert.strictEqual(C.sceneRepId(byId[9]), 9);
  assert.strictEqual(C.sceneRotK(byId[23]), 3);
  assert.strictEqual(C.sceneRepId({ id: 5 }), 5);          // v2.0 manifest entry: no scene fields
  assert.ok(!C.isCopy({ id: 5 }));
  assert.strictEqual(C.sceneRotK({ id: 5 }), 0);
});

t('filterIds: "scenes" = 76 reps (default filter); "todo" never lists copies; v2.0 manifests unchanged', () => {
  const recs = {};
  recs['9'] = rec(byId[9], 40, 30, { done: true });
  recs['0'] = rec(byId[0], 4, 4);                      // edited copy, not done
  recs['94'] = rec(byId[94], 50, 50, { done: true });  // a copy marked done after "Edit anyway"
  const scenes = C.filterIds(images, recs, 'scenes', iaa, true);
  assert.strictEqual(scenes.length, 76);
  assert.ok(scenes.every((id) => byId[id].is_scene_rep));
  const todo = C.filterIds(images, recs, 'todo', iaa, true);
  assert.strictEqual(todo.length, 75);
  assert.ok(!todo.includes(9) && !todo.includes(0));
  assert.deepStrictEqual(C.filterIds(images, recs, 'done', iaa, true), [9, 94]);
  assert.strictEqual(C.filterIds(images, recs, 'all', iaa, true).length, 150);
  assert.deepStrictEqual(C.filterIds(images, recs, 'iaa', iaa, true), manifest.iaa_subset);
  // without scene support every filter behaves as in v2 ('scenes' == all, 'todo' includes copies)
  const plain = images.map((e) => ({ id: e.id }));
  assert.strictEqual(C.filterIds(plain, recs, 'scenes', iaa, false).length, 150);
  assert.strictEqual(C.filterIds(plain, recs, 'todo', iaa, false).length, 148);
});

t('progress counts scenes (n / 76); a done copy does not count; IAA progress over the subset', () => {
  const recs = {};
  recs['9'] = rec(byId[9], 40, 30, { done: true });
  recs['94'] = rec(byId[94], 50, 50, { done: true });
  const i0 = manifest.iaa_subset[0];
  recs[String(i0)] = rec(byId[i0], 10, 10, { done: true });
  const p = C.progress(images, recs, iaa, true);
  assert.deepStrictEqual(p, { unit: 'scenes', done: 2, total: 76, iaaDone: 1, iaaTotal: 20 });
  const q = C.progress(images, recs, iaa, false);
  assert.deepStrictEqual(q, { unit: 'images', done: 3, total: 150, iaaDone: 1, iaaTotal: 20 });
});

t('copiedLabel: identity for k = 0 / 2, h <-> v and warp_dir swapped for k = 1 / 3, rotated lines use TPI', () => {
  const r = rec(byId[45], 40, 30, { warp_dir: 'vertical', weave: 'twill_2_1', done: true });
  const k0 = C.copiedLabel(r, byId[146]);             // png copy, k = 0
  assert.deepStrictEqual([k0.h, k0.v, k0.warp_dir, k0.swap, k0.k], [40, 30, 'vertical', false, 0]);
  const k3 = C.copiedLabel(r, byId[23]);              // quarter-res copy, k = 3
  assert.deepStrictEqual([k3.h, k3.v, k3.h_count, k3.v_count, k3.warp_dir, k3.weave, k3.swap], [30, 40, 30, 40, 'horizontal', 'twill_2_1', true]);
  const k1 = C.copiedLabel(r, byId[138]);             // notebook file, k = 1
  assert.deepStrictEqual([k1.h, k1.v, k1.swap], [30, 40, true]);
  const r2 = rec(byId[67], 20, 10);
  const k2 = C.copiedLabel(r2, byId[63]);             // quarter-res copy, k = 2 (180 deg)
  assert.deepStrictEqual([k2.h, k2.v, k2.swap], [20, 10, false]);
  // a rotated guide line: the copied value is the normalised TPI, not the dot count
  const rr = rec(byId[45], 40, 30);
  C.setAngle(rr, 'v', 20, 'key');
  const L = C.lineInfo(rr, 'v');
  assert.ok(Math.abs(C.copiedLabel(rr, byId[23]).h - L.tpi) < 1e-9);
  assert.strictEqual(C.copiedLabel(null, byId[23]), null);
  assert.strictEqual(C.copiedLabel(r, byId[45]).swap, false);   // the rep itself
});

t('syncIaa: an untouched record follows a changed IAA membership; edited records never change', () => {
  const img = byId[manifest.iaa_subset[0]];
  // opened before the re-draw as a non-IAA image: proposals shown, nothing edited
  const r = C.newRecord(img, { annotator: 'a', blind: false, iaa: false, now: '2026-10-07T10:00:00.000Z' });
  r.time_spent_s = 12;
  assert.ok(C.isPristine(r) && r.proposals_seen && r.h_dots.length > 0);
  assert.ok(C.syncIaa(r, img, true, { policy: 'iaa' }));
  assert.ok(r.iaa && r.blind_mode && r.h_dots.length === 0 && r.v_dots.length === 0);
  assert.ok(r.proposals_seen, 'proposals were seen before: stays true (not blind-clean)');
  assert.strictEqual(r.blind_initial, false);
  assert.strictEqual(r.created, '2026-10-07T10:00:00.000Z');
  assert.strictEqual(r.time_spent_s, 12);
  assert.strictEqual(r.blind_log.length, 1);
  assert.ok(r.blind_log[0].iaa_resync && r.blind_log[0].on);
  assert.ok(C.isPristine(r));
  assert.ok(!C.syncIaa(r, img, true, { policy: 'iaa' }), 'same membership: no-op');
  // edited record: untouched
  const e = C.newRecord(img, { blind: false, iaa: false });
  C.addDot(e, 'h', 3);
  const before = JSON.stringify(e);
  assert.ok(!C.syncIaa(e, img, true, { policy: 'iaa' }));
  assert.strictEqual(JSON.stringify(e), before);
  // dropped from the subset (blind, untouched) -> re-opened with proposals under the 'iaa' policy
  const b = C.newRecord(img, { blind: true, iaa: true });
  assert.ok(C.syncIaa(b, img, false, { policy: 'iaa' }));
  assert.ok(!b.iaa && !b.blind_mode && b.proposals_seen && b.h_dots.length > 0);
});

t('record format unchanged: same keys as v2.0 records and the same CSV columns (+ view_rot_deg appended in 2.2.1)', () => {
  const r = C.newRecord(byId[9], { annotator: 'a' });
  assert.deepStrictEqual(Object.keys(r).sort(), [
    'angle_log', 'annotator', 'blind_initial', 'blind_log', 'blind_mode', 'confidence', 'created', 'done', 'flags',
    'h_angle_deg', 'h_dots', 'h_line_y', 'h_orig', 'h_raw_count', 'h_seg_len_in', 'h_seg_len_px', 'h_tpi', 'iaa', 'id',
    'n_edits', 'n_prop_h', 'n_prop_v', 'nonwoven', 'notes', 'proposal_method', 'proposal_sig', 'proposals_seen',
    'rect_size', 'schema', 'time_spent_s', 'timestamp', 'tool_version', 'v_angle_deg', 'v_dots', 'v_line_x', 'v_orig',
    'v_raw_count', 'v_seg_len_in', 'v_seg_len_px', 'v_tpi', 'warp_dir', 'weave',
  ]);
  assert.strictEqual(C.CSV_COLS_V1.length, 47);
  // v2.2.1 appended one column at the END (view_rot_deg); the first 62 are unchanged
  assert.strictEqual(C.CSV_COLS.length, 63);
  assert.strictEqual(C.CSV_COLS[62], 'view_rot_deg');
  assert.strictEqual(C.SCHEMA, 1);
});

t('isFlagged moved to core.js with the same rule', () => {
  assert.ok(!C.isFlagged(null));
  const r = C.newRecord(byId[9], {});
  assert.ok(!C.isFlagged(r));
  r.confidence = 1; assert.ok(C.isFlagged(r)); r.confidence = 3;
  r.notes = '  '; assert.ok(!C.isFlagged(r));
  r.notes = 'glare'; assert.ok(C.isFlagged(r)); r.notes = '';
  r.flags.blur = true; assert.ok(C.isFlagged(r));
});

console.log(`\nall ${n} scene tests passed`);
