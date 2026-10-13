#!/usr/bin/env node
/**
 * label_day_merge.js -- step 1 helper of tools/label_day.py (reads the exports, never modifies them).
 *
 * Merges every annotation export (tool JSON "Download" / disk autosave, or the tool's CSV) PER ANNOTATOR with
 * the annotation tool's own core.js (parseImport / parseImportCSV / mergeRecords: an edited record beats an
 * untouched one, else the newer timestamp wins), so one person's several sessions (autosave + downloads) become
 * one record set, and two people never collapse into one.
 *
 *   node label_day_merge.js --out DIR [--manifest manifest.json] [--rename OLD=NEW ...] FILE [FILE ...]
 *
 * Writes (DIR is created):
 *   DIR/full/annotations_<name>.json|.csv     every merged record of <name> (exportJSON / exportCSV of core.js)
 *   DIR/scoring/annotations_<name>.json|.csv  the same minus records with a CLEARED line (a line with no counted
 *                                             yarn inside the window = "yarns impossible to see"): that count is
 *                                             missing, and -- as in propagate_scene_labels.py and cnn/train_cv.py --
 *                                             the record is not used for scoring; also minus finished NONWOVEN records
 *                                             (Job V: every scorer drops them, but analyze.py's anchoring and agreement
 *                                             would count the dots left on them, e.g. untouched proposals = 0 error)
 *   DIR/merge_report.json                     per file / per annotator: records, done, conflicts, cleared lines,
 *                                             parse errors, manifest stamp
 * An empty annotator name is reported as "anonymous" (rename it with --rename anonymous=NAME).
 */
'use strict';
const fs = require('fs');
const path = require('path');

const TOOL = path.resolve(__dirname, '..', '..', 'annotation_tool');
const C = require(path.join(TOOL, 'core.js'));

function parseArgs(argv) {
  const a = { out: null, manifest: path.join(TOOL, 'data', 'manifest.json'), rename: {}, files: [] };
  for (let i = 0; i < argv.length; i++) {
    const t = argv[i];
    if (t === '--out') a.out = argv[++i];
    else if (t === '--manifest') a.manifest = argv[++i];
    else if (t === '--rename') {
      const s = argv[++i] || '';
      const k = s.indexOf('=');
      if (k < 0) throw new Error(`--rename needs OLD=NEW, got "${s}"`);
      a.rename[s.slice(0, k).trim() || 'anonymous'] = s.slice(k + 1).trim();
    } else a.files.push(t);
  }
  if (!a.out || !a.files.length) throw new Error('usage: node label_day_merge.js --out DIR [--rename OLD=NEW] FILE...');
  return a;
}

function lineMissing(r, line) {
  const d = C.derived(r);
  const t = d[line + '_tpi'];
  return !(typeof t === 'number' && isFinite(t) && t > 0);
}

function main() {
  const a = parseArgs(process.argv.slice(2));
  const man = JSON.parse(fs.readFileSync(a.manifest, 'utf8'));
  const stores = {};          // name -> {name: rec}
  const fileRep = [];
  const conflicts = {};       // name -> [{id, kept, dropped}]
  a.files.forEach((f) => {
    const fr = { path: path.resolve(f), kind: /\.json$/i.test(f) ? 'json' : 'csv', annotators: {}, n_records: 0, n_done: 0 };
    let imp;
    try {
      const text = fs.readFileSync(f, 'utf8');
      if (fr.kind === 'json') {
        const obj = JSON.parse(text);
        imp = C.parseImport(obj);
        fr.manifest_generated = obj.manifest_generated || null;
        fr.tool_version = obj.tool_version || null;
        fr.exported = obj.exported || null;
      } else {
        imp = C.parseImportCSV(text);
      }
    } catch (e) {
      fr.error = String(e && e.message ? e.message : e);
      fileRep.push(fr);
      return;
    }
    Object.keys(imp.records).forEach((k) => {
      const r = imp.records[k];
      let name = String(r.annotator || imp.annotator || '').trim() || 'anonymous';
      if (Object.prototype.hasOwnProperty.call(a.rename, name)) name = a.rename[name];
      r.annotator = name === 'anonymous' ? '' : name;
      fr.n_records++;
      if (r.done) fr.n_done++;
      fr.annotators[name] = (fr.annotators[name] || 0) + 1;
      const base = stores[name] || {};
      const prev = base[k];
      const m = C.mergeRecords(base, { [k]: r });
      if (prev) {
        const won = m.merged[k];
        const lost = won === r ? prev : r;
        const same = lost.h_dots.length === won.h_dots.length && lost.v_dots.length === won.v_dots.length &&
          !!lost.done === !!won.done && (lost.weave || null) === (won.weave || null);
        if (!same) {
          (conflicts[name] = conflicts[name] || []).push({
            id: Number(k), kept_timestamp: won.timestamp || null, dropped_timestamp: lost.timestamp || null,
            kept_counts: [won.h_dots.length, won.v_dots.length], dropped_counts: [lost.h_dots.length, lost.v_dots.length],
            kept_done: !!won.done, dropped_done: !!lost.done, file: fr.path,
          });
        }
      }
      stores[name] = m.merged;
    });
    fileRep.push(fr);
  });
  const ann = {};
  ['full', 'scoring'].forEach((d) => fs.mkdirSync(path.join(a.out, d), { recursive: true }));
  Object.keys(stores).sort().forEach((name) => {
    const recs = stores[name];
    const st = C.emptyStore(name === 'anonymous' ? '' : name);
    st.records = recs;
    const safe = C.safeName(name);
    fs.writeFileSync(path.join(a.out, 'full', `annotations_${safe}.json`), JSON.stringify(C.exportJSON(st, man), null, 1));
    fs.writeFileSync(path.join(a.out, 'full', `annotations_${safe}.csv`), C.exportCSV(st));
    const cleared = [];
    const keep = {};
    Object.keys(recs).forEach((k) => {
      const r = recs[k];
      const miss = ['h', 'v'].filter((l) => lineMissing(r, l));
      if (miss.length && r.done && !r.nonwoven) cleared.push({ id: Number(k), lines: miss });
      if (!r.done || (!miss.length && !r.nonwoven)) keep[k] = r;
    });
    const sc = C.emptyStore(st.annotator);
    sc.records = keep;
    fs.writeFileSync(path.join(a.out, 'scoring', `annotations_${safe}.json`), JSON.stringify(C.exportJSON(sc, man), null, 1));
    fs.writeFileSync(path.join(a.out, 'scoring', `annotations_${safe}.csv`), C.exportCSV(sc));
    const vals = Object.values(recs);
    ann[name] = {
      file_stem: `annotations_${safe}`, n_records: vals.length, n_done: vals.filter((r) => r.done).length,
      n_csv_imported: vals.filter((r) => r.imported_from_csv).length,
      csv_imported_ids: vals.filter((r) => r.imported_from_csv).map((r) => Number(r.id)).sort((x, y) => x - y),
      cleared, conflicts: conflicts[name] || [],
      tool_versions: Array.from(new Set(vals.map((r) => r.tool_version || 'unknown'))).sort(),
    };
  });
  const rep = { written: new Date().toISOString(), manifest: path.resolve(a.manifest), manifest_generated: man.generated || null,
    rename: a.rename, files: fileRep, annotators: ann };
  fs.writeFileSync(path.join(a.out, 'merge_report.json'), JSON.stringify(rep, null, 1));
  const nOk = fileRep.filter((f) => !f.error).length;
  console.log(`merged ${nOk}/${fileRep.length} file(s) -> ${Object.keys(ann).length} annotator(s): ` +
    Object.keys(ann).map((n) => `${n} (${ann[n].n_done} done / ${ann[n].n_records})`).join(', '));
}

try { main(); } catch (e) { console.error('label_day_merge.js:', e && e.message ? e.message : e); process.exit(2); }
