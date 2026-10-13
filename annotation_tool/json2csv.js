#!/usr/bin/env node
/**
 * Convert annotation JSON (the Download export, or the server autosave
 * data/saved/annotations_<name>.json) to the same CSV the tool's Download writes, using
 * core.js itself, so score150.py / quality150.py / cnn/train_cv.py can always be run even if
 * an annotator only has the JSON.
 *
 *   node json2csv.js data/saved/annotations_A.json [more.json ...]
 *   -> writes data/saved/annotations_A.csv next to each input (use --out-dir DIR to change)
 */
'use strict';
const fs = require('fs');
const path = require('path');
const C = require('./core.js');

const args = process.argv.slice(2);
let outDir = null;
const files = [];
for (let i = 0; i < args.length; i++) {
  if (args[i] === '--out-dir') outDir = args[++i];
  else files.push(args[i]);
}
if (!files.length) {
  console.error('usage: node json2csv.js FILE.json [FILE2.json ...] [--out-dir DIR]');
  process.exit(2);
}
files.forEach((f) => {
  const imp = C.parseImport(JSON.parse(fs.readFileSync(f, 'utf8')));
  Object.values(imp.records).forEach((r) => { if (!r.annotator) r.annotator = imp.annotator; });
  const store = C.emptyStore(imp.annotator);
  store.records = imp.records;
  const out = path.join(outDir || path.dirname(f), path.basename(f).replace(/\.json$/i, '') + '.csv');
  fs.writeFileSync(out, C.exportCSV(store));
  const n = Object.keys(imp.records).length;
  const done = Object.values(imp.records).filter((r) => r.done).length;
  console.log(`${f} -> ${out}  (${n} records, ${done} done, annotator "${imp.annotator}")`);
});
