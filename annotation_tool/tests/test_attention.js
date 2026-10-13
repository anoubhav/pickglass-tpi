// Headless tests of the v2.2 "check carefully" attention logic (core.js) and of the attention fields
// in data/manifest.json:
//   node tests/test_attention.js
// No labels are used: every record here is a fresh in-memory record (never saved).
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
const reps = images.filter((e) => e.is_scene_rep);
let n = 0;
const t = (name, fn) => { fn(); n++; console.log('ok', n, name); };

const FIXED = new Set([
  'very fine yarns for this resolution: zoom in well', 'fine yarns for this resolution: zoom in',
  'the photo may be too soft to resolve every yarn (blur)', 'glare hides part of the window: move the line to a clear row',
  'the automatic counter is unsure about this photo',
]);
const DIS = /^estimators disagree on [HV] by \d+%$/;
const SIGNAL_KEYS = ['conf', 'd_h', 'd_v', 'estimators', 'eta', 'n_h', 'n_v', 'ppy_bin', 'sat240'];
// every number shown must be a percentage or the "(+n more)" suffix: never a count
function countFree(text) {
  const rest = String(text).replace(/\d+%/g, '').replace(/\(\+\d+ more\)$/, '');
  return !/\d/.test(rest);
}

t('manifest: attention on exactly the 76 reps; copies have none; block counts match', () => {
  assert.ok(C.hasAttention(manifest));
  assert.strictEqual(reps.length, 76);
  reps.forEach((e) => {
    assert.ok(['low', 'medium', 'high'].includes(e.attention), 'level of ' + e.id);
    assert.ok(Number.isInteger(e.attention_score) && e.attention_score >= 0);
    assert.strictEqual(e.attention, e.attention_score >= 2 ? 'high' : e.attention_score === 1 ? 'medium' : 'low');
    assert.ok(Array.isArray(e.attention_reasons) && Array.isArray(e.attention_codes));
    assert.strictEqual(e.attention_reasons.length, e.attention_codes.length);
    assert.strictEqual(e.attention_reasons.length === 0, e.attention === 'low');
  });
  images.filter((e) => !e.is_scene_rep).forEach((e) => {
    ['attention', 'attention_score', 'attention_reasons', 'attention_codes', 'attention_signals']
      .forEach((k) => assert.ok(!(k in e), k + ' on copy ' + e.id));
  });
  const cnt = { low: 0, medium: 0, high: 0 }, out = { low: 0, medium: 0, high: 0 };
  reps.forEach((e) => { cnt[e.attention]++; if (!iaa.has(e.id)) out[e.attention]++; });
  assert.deepStrictEqual(manifest.attention.levels, cnt);
  assert.deepStrictEqual(manifest.attention.levels_outside_iaa, out);
  assert.strictEqual(manifest.attention.n_reps, 76);
});

t('manifest: reasons are generic and count-free; signals hold no count', () => {
  reps.forEach((e) => {
    e.attention_reasons.forEach((r) => {
      assert.ok(DIS.test(r) || FIXED.has(r), 'unexpected reason on ' + e.id + ': ' + r);
      assert.ok(countFree(r), r);
    });
    e.attention_codes.forEach((c, i) => {
      if (c.startsWith('disagree_')) assert.ok(e.attention_reasons[i].includes(' on ' + c.slice(-1).toUpperCase() + ' '));
    });
    const s = e.attention_signals;
    assert.deepStrictEqual(Object.keys(s).sort(), SIGNAL_KEYS);
    assert.ok([null, '<4', '4-5', '>=5'].includes(s.ppy_bin));
    // the disagreement reasons quote the stored D (rounded) and need D >= 10 %
    ['h', 'v'].forEach((ax) => {
      const r = e.attention_reasons.find((x) => x.startsWith('estimators disagree on ' + ax.toUpperCase()));
      if (r) assert.strictEqual(Number(r.match(/(\d+)%/)[1]), Math.round(s['d_' + ax]));
      else assert.ok(s['d_' + ax] == null || s['d_' + ax] < 10, 'D >= 10% without a reason on ' + e.id);
    });
  });
  // the manifest does not even carry an estimator count next to the attention fields
  const blob = JSON.stringify(reps.map((e) => [e.attention_reasons, e.attention_signals]));
  ['h_count', 'v_count', 'vertical', 'horizontal', 'h_gabor', 'pred_h', 'consensus'].forEach((k) => assert.ok(!blob.includes(k), k));
});

t('attentionVisible: never on IAA-subset scenes, copies or blind records', () => {
  const hi = reps.find((e) => e.attention === 'high' && !iaa.has(e.id));
  const hiIaa = reps.find((e) => e.attention === 'high' && iaa.has(e.id));
  assert.ok(hi && hiIaa);
  const open = C.newRecord(hi, { blind: false, iaa: false });
  assert.ok(C.attentionVisible(hi, open, iaa));
  assert.ok(!C.attentionVisible(hi, C.newRecord(hi, { blind: true, iaa: false }), iaa), 'blind record');
  // an IAA scene: hidden whatever the record says, also if blind was turned off
  [null, C.newRecord(hiIaa, { blind: true, iaa: true }), C.newRecord(hiIaa, { blind: false, iaa: true })].forEach((r) => {
    assert.ok(!C.attentionVisible(hiIaa, r, iaa));
    ['iaa', 'none', 'manual', 'random', 'all'].forEach((p) => assert.ok(!C.attentionVisible(hiIaa, r, iaa, { policy: p })));
  });
  // copies (also of a flagged scene, also when unlocked) never show it
  images.filter((e) => !e.is_scene_rep).forEach((e) => {
    assert.ok(!C.attentionVisible(e, null, iaa));
    assert.ok(!C.attentionVisible(Object.assign({}, e, { attention: 'high', attention_reasons: ['x'] }),
      C.newRecord(e, { blind: false }), iaa));
  });
  // a copy pretending to be a rep of an IAA scene: still hidden through its scene id
  const fake = Object.assign({}, hi, { id: 9999, scene_id: hiIaa.id, is_scene_rep: true });
  assert.ok(!C.attentionVisible(fake, null, iaa));
});

t('attentionVisible: an image not opened yet follows the blind policy it will open with', () => {
  const hi = reps.find((e) => e.attention === 'high' && !iaa.has(e.id));
  assert.ok(C.attentionVisible(hi, null, iaa));                        // default 'iaa'
  assert.ok(C.attentionVisible(hi, null, iaa, { policy: 'none' }));
  assert.ok(!C.attentionVisible(hi, null, iaa, { policy: 'all' }));
  assert.ok(!C.attentionVisible(hi, null, iaa, { policy: 'manual', manualDefault: true }));
  assert.ok(C.attentionVisible(hi, null, iaa, { policy: 'manual', manualDefault: false }));
  reps.filter((e) => !iaa.has(e.id)).forEach((e) => {
    assert.strictEqual(C.attentionVisible(e, null, iaa, { policy: 'random' }), !C.hashBlind(e.id));
  });
});

t('attentionDisplay: nothing for low; "Check carefully" / "Take care" + first reason; never a count', () => {
  reps.forEach((e) => {
    const d = C.attentionDisplay(e);
    if (e.attention === 'low') { assert.strictEqual(d, null); return; }
    assert.strictEqual(d.level, e.attention);
    const pre = e.attention === 'high' ? 'Check carefully: ' : 'Take care: ';
    assert.ok(d.badge.startsWith(pre + e.attention_reasons[0]), d.badge);
    if (e.attention_reasons.length > 1) assert.ok(d.badge.endsWith('(+' + (e.attention_reasons.length - 1) + ' more)'));
    assert.strictEqual(d.line, e.attention_reasons.join(' · '));
    assert.ok(countFree(d.badge) && countFree(d.line), d.badge);
    // no proposal count (what ours put on the line) appears in what is shown
    const nums = (d.badge + ' ' + d.line).match(/\d+(?!\d*%)/g) || [];
    nums.forEach((x) => assert.ok(/\(\+\d+ more\)/.test(d.badge) && d.badge.includes('(+' + x + ' more)'), 'number ' + x));
  });
  assert.strictEqual(C.attentionDisplay({ id: 1 }), null);
  assert.strictEqual(C.attentionDisplay({ id: 1, attention: 'bogus', attention_reasons: ['a'] }), null);
  const d = C.attentionDisplay({ id: 1, attention: 'medium', attention_reasons: [7, '', ' glare '] });
  assert.strictEqual(d.badge, 'Take care: glare');   // non-strings and blanks are dropped
  assert.ok(C.attentionDisplay({ id: 1, attention: 'high' }).badge.startsWith('Check carefully: '));
});

t('filter "attention": visible reps only, high -> medium -> low, then score, then id', () => {
  const ids = C.filterIds(images, {}, 'attention', iaa, true, { policy: 'iaa' });
  const expected = reps.filter((e) => !iaa.has(e.id));
  assert.strictEqual(ids.length, expected.length);
  assert.deepStrictEqual(ids.slice().sort((a, b) => a - b), expected.map((e) => e.id));
  ids.forEach((id) => assert.ok(!iaa.has(id) && byId[id].is_scene_rep));
  const R = { low: 0, medium: 1, high: 2 };
  for (let i = 1; i < ids.length; i++) {
    const a = byId[ids[i - 1]], b = byId[ids[i]];
    const ka = [-R[a.attention], -a.attention_score, a.id], kb = [-R[b.attention], -b.attention_score, b.id];
    assert.ok(ka[0] < kb[0] || (ka[0] === kb[0] && (ka[1] < kb[1] || (ka[1] === kb[1] && ka[2] < kb[2]))), 'order at ' + i);
  }
  assert.strictEqual(byId[ids[0]].attention, 'high');
  // a non-IAA rep whose record is blind drops out; done records stay
  const recs = {};
  recs[String(ids[0])] = C.newRecord(byId[ids[0]], { blind: true });
  const done = C.newRecord(byId[ids[1]], { blind: false }); done.done = true; recs[String(ids[1])] = done;
  const ids2 = C.filterIds(images, recs, 'attention', iaa, true, { policy: 'iaa' });
  assert.ok(!ids2.includes(ids[0]) && ids2.includes(ids[1]) && ids2.length === ids.length - 1);
  // "always blind" policy: nothing is ranked until an image is opened non-blind
  assert.deepStrictEqual(C.filterIds(images, {}, 'attention', iaa, true, { policy: 'all' }), []);
  // the other filters are unchanged by the extra argument
  ['scenes', 'all', 'todo', 'done', 'iaa', 'flagged'].forEach((f) => {
    assert.deepStrictEqual(C.filterIds(images, recs, f, iaa, true, { policy: 'all' }), C.filterIds(images, recs, f, iaa, true));
  });
});

t('a manifest without attention fields behaves exactly like v2.1', () => {
  const strip = (e) => {
    const o = Object.assign({}, e);
    ['attention', 'attention_score', 'attention_reasons', 'attention_codes', 'attention_signals'].forEach((k) => delete o[k]);
    return o;
  };
  const old = images.map(strip);
  assert.ok(!C.hasAttention({ images: old }));
  assert.ok(!C.hasAttention({}));
  assert.deepStrictEqual(C.filterIds(old, {}, 'attention', iaa, true), []);
  old.forEach((e) => { assert.ok(!C.attentionVisible(e, null, iaa)); assert.strictEqual(C.attentionDisplay(e), null); });
  ['scenes', 'todo', 'iaa'].forEach((f) => assert.deepStrictEqual(C.filterIds(old, {}, f, iaa, true), C.filterIds(images, {}, f, iaa, true)));
});

t('record format unchanged (same keys; the 62 CSV columns + view_rot_deg appended in 2.2.1); tool_version 2.2.x', () => {
  const r = C.newRecord(reps[0], { annotator: 'a' });
  assert.ok(!Object.keys(r).some((k) => k.startsWith('attention')));
  assert.strictEqual(Object.keys(r).length, 42);
  // v2.2.1 appended view_rot_deg at the END; the first 62 columns are unchanged
  assert.strictEqual(C.CSV_COLS.length, 63);
  assert.strictEqual(C.CSV_COLS[62], 'view_rot_deg');
  assert.ok(!C.CSV_COLS.some((k) => k.startsWith('attention')));
  // still 2.2.x: paper/tools/label_day.py finds badge-era records with tool_version.startswith("2.2")
  assert.ok(/^2\.2\.\d+$/.test(C.TOOL_VERSION), C.TOOL_VERSION);
  assert.strictEqual(r.tool_version, C.TOOL_VERSION);
});

t('app.js shows attention only through attentionVisible() and never reads the signals', () => {
  const src = fs.readFileSync(path.join(ROOT, 'app.js'), 'utf8');
  const uses = src.split('\n').filter((l) => l.includes('attentionDisplay('));
  assert.strictEqual(uses.length, 1);
  assert.ok(uses[0].includes('C.attentionVisible(img, r, st.iaa, blindOpts())') && uses[0].includes('!st.ephemeral'));
  ['attention_signals', 'attention_codes', 'attention_score', '.textContent = attn.line', 'innerHTML = attn']
    .forEach((k) => assert.ok(!src.includes(k), k));
  const html = fs.readFileSync(path.join(ROOT, 'index.html'), 'utf8');
  assert.ok(html.includes('id="attnbadge"') && html.includes('value="attention"'));
});

console.log(`\nall ${n} attention tests passed`);
