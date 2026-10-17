#!/usr/bin/env python
"""Tests for propagate_labels.py (and its agreement with annotation_tool/core.js).

    <venv2>/bin/python tests/test_propagate.py

1. Python port of core.js lineInfo().tpi == core.js derived().h_tpi / v_tpi on 300 random
   v1/v2 records (angles, off-window dots, odd line positions)        [needs node]
2. JSON (v2 export shape) and the CSV that core.js exportCSV() writes for the same records
   read back to the same labels                                          [needs node]
3. Dedup: same annotator + image in two files -> newest timestamp wins; done / nonwoven filters
4. Swap: k = 90/270 rows have h <-> v, angles swapped, warp_dir swapped; k = 0/180 unchanged;
   group == source_id; only usable manifest rows; review decisions applied
5. Scenes (CLI default): a label on a scene's rep reaches the augmentations of every
   file of the scene, rotated twice (rep -> file by the file's scene_rot_k, file -> aug by k_deg);
   group == scene_id; a copy's label fills an unlabelled rep's scene; --no-scenes = test 4 behaviour
Fixture files are (over)written to tests/fixtures/ (never deleted).
"""
from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import propagate_labels as P  # noqa: E402

CORE = os.path.join(P.ROOT, "annotation_tool", "core.js")
FIX = os.path.join(HERE, "fixtures")
NODE = shutil.which("node")


def rand_record(rng, i, annotator="T", ts="2026-10-07T10:00:00.000Z", done=True):
    S = rng.randint(300, 2000)
    rec = dict(schema=1, id=i, annotator=annotator, rect_size=S, h_line_y=round(rng.uniform(0, S), 1),
               v_line_x=round(rng.uniform(0, S), 1),
               h_dots=sorted(round(rng.uniform(-10, S + 10), 1) for _ in range(rng.randint(5, 90))),
               v_dots=sorted(round(rng.uniform(-10, S + 10), 1) for _ in range(rng.randint(5, 90))),
               weave=rng.choice(["plain", "twill_2_1", "satin", "unsure"]),
               warp_dir=rng.choice(["vertical", "horizontal", "unknown", None]), done=done, nonwoven=False,
               confidence=rng.choice([1, 2, 3, None]), flags={}, timestamp=ts, blind_log=[])
    if rng.random() < 0.7:   # v2 angles (some exactly 0, some at the clamp)
        rec["h_angle_deg"] = rng.choice([0, round(rng.uniform(-45, 45), 2), 45, -50])
        rec["v_angle_deg"] = rng.choice([0, round(rng.uniform(-45, 45), 2), -45])
    rec["h_orig"] = [0] * len(rec["h_dots"])
    rec["v_orig"] = [0] * len(rec["v_dots"])
    return rec


def node_eval(js):
    return subprocess.run([NODE, "-e", js], capture_output=True, text=True, check=True).stdout


def test_port_and_roundtrip():
    if not NODE:
        print("SKIP 1-2: node not found")
        return
    rng = random.Random(0)
    recs = [rand_record(rng, i) for i in range(300)]
    os.makedirs(FIX, exist_ok=True)
    jpath = os.path.join(FIX, "export_v2.json")
    json.dump({"schema": 1, "annotator": "T", "records": {str(r["id"]): r for r in recs}}, open(jpath, "w"))
    cpath = os.path.join(FIX, "export_v2.csv")
    out = node_eval(f"""
const C = require({json.dumps(CORE)}); const fs = require('fs');
const imp = C.parseImport(JSON.parse(fs.readFileSync({json.dumps(jpath)}, 'utf8')));
const st = C.emptyStore('T'); st.records = imp.records;
fs.writeFileSync({json.dumps(cpath)}, C.exportCSV(st));
const d = {{}}; Object.values(imp.records).forEach(r => {{ const x = C.derived(r); d[r.id] = [x.h_tpi, x.v_tpi]; }});
process.stdout.write(JSON.stringify(d));""")
    ref = json.loads(out)
    bad = 0
    for r in recs:
        for q, line in enumerate("hv"):
            py = P.line_tpi(r, line)
            js = ref[str(r["id"])][q]
            if (py is None) != (js is None) or (py is not None and abs(round(py, 2) - js) > 0.011):
                bad += 1
                print("MISMATCH", r["id"], line, py, js)
    assert bad == 0, f"{bad} tpi mismatches vs core.js"
    a = pd.DataFrame(P.read_json(jpath)).set_index("source_id").sort_index()
    b = pd.DataFrame(P.read_csv(cpath)).set_index("source_id").sort_index()
    assert np.allclose(a.h_tpi.round(2), b.h_tpi, atol=0.011) and np.allclose(a.v_tpi.round(2), b.v_tpi, atol=0.011)
    assert (a.weave.fillna("") == b.weave.fillna("")).all()
    assert (a.warp_dir.fillna("") == b.warp_dir.fillna("")).all()
    print(f"OK 1-2: lineInfo port == core.js on {2 * len(recs)} lines; JSON and CSV exports read identically")


def test_dedup_and_filters():
    rng = random.Random(1)
    old = rand_record(rng, 5, ts="2026-10-07T09:00:00.000Z")
    new = dict(old, h_dots=old["h_dots"][:10], h_angle_deg=0, timestamp="2026-10-07T11:00:00.000Z")
    nd = rand_record(rng, 6, done=False)
    nw = dict(rand_record(rng, 7), nonwoven=True)
    os.makedirs(FIX, exist_ok=True)
    p1, p2 = os.path.join(FIX, "dedup_old.json"), os.path.join(FIX, "dedup_new.json")
    json.dump({"annotator": "T", "records": [old, nd, nw]}, open(p1, "w"))
    json.dump({"annotator": "T", "records": [new]}, open(p2, "w"))
    src, _ = P.read_annotations([p1, p2], log=lambda *a: None)
    assert list(src.source_id) == [5], src
    assert src.h_tpi.iloc[0] == 10, "newest timestamp must win"
    src2, _ = P.read_annotations([p1, p2], include_not_done=True, log=lambda *a: None)
    assert sorted(src2.source_id) == [5, 6]
    # two annotators -> mean
    other = dict(new, annotator="U", h_dots=new["h_dots"][:6] + [1, 2, 3, 4, 5, 6])
    p3 = os.path.join(FIX, "dedup_other.json")
    json.dump({"annotator": "U", "records": [other]}, open(p3, "w"))
    src3, _ = P.read_annotations([p1, p2, p3], log=lambda *a: None)
    assert src3.h_tpi.iloc[0] == 11 and src3.n_annotations.iloc[0] == 2
    print("OK 3: dedup (newest wins), done / nonwoven filters, multi-annotator mean")


def test_swap():
    man_all = pd.read_csv(P.MANIFEST)
    src = pd.DataFrame(dict(source_id=range(150), h_tpi=[10.0 + i for i in range(150)],
                            v_tpi=[1000.0 + i for i in range(150)], h_count=[1.0] * 150, v_count=[2.0] * 150,
                            h_angle_deg=[2.5] * 150, v_angle_deg=[-3.0] * 150, warp_dir=["vertical"] * 150,
                            weave=["twill_2_1"] * 150))
    man = P.load_manifest()
    lab = P.propagate(src, man)
    assert len(lab) == int(man_all.usable.sum()) or os.path.exists(P.DECISIONS)
    assert set(lab.aug_id) <= set(man_all.aug_id[man_all.usable | man_all.aug_id.isin(lab.aug_id)])
    sw = lab.swap_hv.astype(bool)
    assert (lab.k_deg[sw].isin([90, 270])).all() and (lab.k_deg[~sw].isin([0, 180])).all()
    assert np.allclose(lab.h_tpi[~sw], 10 + lab.source_id[~sw]) and np.allclose(lab.v_tpi[~sw], 1000 + lab.source_id[~sw])
    assert np.allclose(lab.h_tpi[sw], 1000 + lab.source_id[sw]) and np.allclose(lab.v_tpi[sw], 10 + lab.source_id[sw])
    assert (lab.h_count[sw] == 2).all() and (lab.v_count[sw] == 1).all()
    assert (lab.h_angle_deg[sw] == -3.0).all() and (lab.v_angle_deg[sw] == 2.5).all()
    assert (lab.warp_dir[sw] == "horizontal").all() and (lab.warp_dir[~sw] == "vertical").all()
    assert (lab.weave == "twill_2_1").all()
    assert (lab.group == lab.source_id).all()
    assert lab.image.map(os.path.exists).all()
    bad = man_all[~man_all.usable].aug_id
    dec = pd.read_csv(P.DECISIONS, dtype=str, keep_default_na=False) if os.path.exists(P.DECISIONS) else None
    accepted = set(dec.aug_id[dec.decision.str.lower() == "accept"]) if dec is not None else set()
    assert not (set(bad) - accepted) & set(lab.aug_id), "flagged rows must not be labelled unless accepted"
    print(f"OK 4: {len(lab)} rows, {int(sw.sum())} swapped; flagged rows excluded; group == source_id")


def test_scenes():
    sc = P.load_scenes(log=lambda *a: None)
    if sc is None:
        print("SKIP 5: the annotation-tool manifest has no scene fields")
        return
    os.makedirs(FIX, exist_ok=True)
    S = 500

    def rec(i, nh, nv, warp):
        return dict(schema=1, id=i, annotator="T", rect_size=S, h_line_y=250, v_line_x=250,
                    h_dots=[float(x) for x in range(nh)], v_dots=[float(x) for x in range(nv)], weave="plain",
                    warp_dir=warp, done=True, nonwoven=False, timestamp="2026-10-09T00:00:00Z")
    # scene 45 = [23, 37, 45, 138, 146] labelled on its rep; scene 9 = [0, 9] labelled only on copy 0
    jp = os.path.join(FIX, "scenes_export.json")
    json.dump({"annotator": "T", "records": [rec(45, 27, 43, "vertical"), rec(0, 47, 38, "horizontal")]}, open(jp, "w"))
    out = os.path.join(FIX, "scenes_labels.csv")
    lab = P.run([jp], out=out, log=lambda *a: None)
    man = P.load_manifest()
    for scene, members in ((45, [23, 37, 45, 138, 146]), (9, [0, 9])):
        assert set(lab.source_id[lab.scene_id == scene]) == set(members) & set(man.source_id), (scene, lab.source_id.unique())
    assert (lab.group == lab.scene_id).all()
    base = {45: (27.0, 43.0, "vertical"), 9: (47.0, 38.0, "horizontal")}     # rep frame (0 and 9: k = 0)
    for r in lab.itertuples():
        h, v, w = base[r.scene_id]
        odd = (int(r.source_scene_rot_k) % 2 == 1) != bool(r.swap_hv)            # rep -> file -> aug parity
        exp = (v, h, {"vertical": "horizontal", "horizontal": "vertical"}[w]) if odd else (h, v, w)
        assert (r.h_tpi, r.v_tpi, r.warp_dir) == exp, (r.aug_id, r.h_tpi, r.v_tpi, r.warp_dir, exp)
    assert set(lab.label_src_id[lab.scene_id == 9]) == {0} and set(lab.label_src_id[lab.scene_id == 45]) == {45}
    assert set(lab.source_kind) >= {"rep", "quarter_res_copy", "byte_identical", "notebook_synthetic", "png_copy"}
    legacy = P.run([jp], out=os.path.join(FIX, "scenes_labels_legacy.csv"), scenes=False, group_by="source",
                   log=lambda *a: None)
    assert set(legacy.source_id) == {0, 45} and (legacy.group == legacy.source_id).all()
    assert len(lab) > len(legacy)
    print(f"OK 5: scene labels reach {lab.source_id.nunique()} source files / {len(lab)} augmentations "
          f"(legacy: {legacy.source_id.nunique()} / {len(legacy)}); double rotation parity; group == scene_id")


if __name__ == "__main__":
    test_port_and_roundtrip()
    test_dedup_and_filters()
    test_swap()
    test_scenes()
    print("all tests passed")
