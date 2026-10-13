#!/usr/bin/env python
"""Tests of the scene structure (scenes.py), the IAA v2 draw (prepare.py) and the scene label
propagation (propagate_scene_labels.py), incl. agreement with core.js copiedLabel().

    <venv2>/bin/python tests/test_scenes.py          (node is needed for test 6; skipped without it)

Writes nothing except a temporary directory under the system temp dir (removed by the OS).
"""
from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
TOOL = os.path.dirname(HERE)
sys.path.insert(0, TOOL)
import prepare as PR  # noqa: E402
import propagate_scene_labels as PS  # noqa: E402
import scenes as SC  # noqa: E402

MAN = json.load(open(os.path.join(TOOL, "data", "manifest.json")))
E = {e["id"]: e for e in MAN["images"]}
NODE = shutil.which("node")
n = 0


def t(fn):
    global n
    fn()
    n += 1
    print("ok", n, fn.__doc__.strip().splitlines()[0])


@t
def test_structure():
    """scene_structure: 76 scenes, kinds, rep rule == scenes_to_annotate.txt, copy_of sane"""
    st = SC.scene_structure()
    assert len(st) == 150
    reps = sorted({s["scene_id"] for s in st.values()})
    assert len(reps) == 76 and all(st[r]["is_scene_rep"] for r in reps)
    files, _ = SC.load_provenance()
    for i, s in st.items():
        e = files[i]
        if s["kind"] == "byte_identical":
            assert e["md5"] == files[s["copy_of"]]["md5"]
            assert s["copy_of"] < i or s["copy_of"] == s["scene_id"]
        if s["kind"] == "png_copy":
            assert e["container"] == "PNG" and sorted(e["size"]) == sorted(files[s["copy_of"]]["size"])
        if s["kind"] == "notebook_synthetic":
            assert e["container"] == "PNG" and SC._synth_of(e["size"], files[s["copy_of"]]["size"])
        if s["kind"] == "quarter_res_copy":
            r = SC._px(files[s["scene_id"]]) / SC._px(e)
            assert abs(r - 16) < 0.01, (i, r)          # exactly 4x reduced
        if not s["is_scene_rep"]:
            rep = files[s["scene_id"]]
            assert rep["container"] == "JPEG"
            assert SC._px(rep) >= SC._px(e) or e["container"] == "PNG"
    assert SC.read_reps_file() == {r: st[r]["scene_members"] for r in reps}
    # the manifest carries the same structure
    for i, s in st.items():
        for k in ("scene_id", "is_scene_rep", "scene_members", "kind", "copy_of"):
            assert E[i][k] == s[k], (i, k)


@t
def test_manifest_rotation_fields():
    """manifest rotation: k in 0..3, swap == k odd, reps k = 0, copies inherit their source's k"""
    for i, e in E.items():
        assert e["scene_rot_k"] in (0, 1, 2, 3)
        assert e["scene_swap_hv"] == (e["scene_rot_k"] % 2 == 1)
        if e["is_scene_rep"]:
            assert e["scene_rot_k"] == 0 and e["scene_rot_margin"] is None
        else:
            assert e["scene_rot_margin"] > 0.2, (i, e["scene_rot_margin"])
        if e["kind"] in ("byte_identical", "png_copy"):
            assert e["scene_rot_k"] == E[e["copy_of"]]["scene_rot_k"]
        if e["kind"] == "notebook_synthetic":
            assert (e["scene_rot_k"] - E[e["copy_of"]]["scene_rot_k"]) % 2 == 1
    ks = [e["scene_rot_k"] for e in E.values() if e["kind"] == "quarter_res_copy"]
    assert (ks.count(0), ks.count(1), ks.count(2), ks.count(3)) == (11, 0, 3, 33)
    # independent check written by scenes.py --check-images (if present)
    chk = os.path.join(TOOL, "data", "scenes_check.csv")
    if os.path.exists(chk):
        c = pd.read_csv(chk)
        assert len(c) == 74 and c.agree.all()
        assert all(E[int(r.id)]["scene_rot_k"] == int(r.k_image) for r in c.itertuples())


@t
def test_iaa_v2():
    """IAA v2: deterministic, 20 reps of distinct scenes, proportional strata, forced ids kept"""
    entries = [dict(e) for e in MAN["images"]]
    a, tab = PR.iaa_subset_v2(entries)
    b, _ = PR.iaa_subset_v2(entries)
    assert a == b == MAN["iaa_subset"], (a, MAN["iaa_subset"])
    assert len(a) == 20 and all(E[i]["is_scene_rep"] for i in a) and len({E[i]["scene_id"] for i in a}) == 20
    assert sum(x["n_drawn"] for x in tab) == 20
    for x in tab:
        assert abs(x["n_drawn"] - x["quota"]) < 1
        assert len(x["ids"]) == x["n_drawn"]
    # a done record on an old-subset image is kept (here: 61, a rep in the old subset)
    c, tab2 = PR.iaa_subset_v2(entries, forced=[61])
    assert 61 in c and len(c) == 20 and sum(x["n_drawn"] for x in tab2) == 20
    # choose_iaa: frozen once v2 is written
    iaa, f = PR.choose_iaa(MAN, entries, "now")
    assert iaa == MAN["iaa_subset"] and f["iaa_history"] == MAN["iaa_history"]
    # retiring v1 with an empty saved dir: new subset + a history entry
    v1 = dict(MAN)
    for k in ("iaa_version", "iaa_policy", "iaa_history"):
        v1.pop(k, None)
    v1["iaa_subset"] = MAN["iaa_history"][0]["subset"]
    old_saved = PR.SAVED
    PR.SAVED = tempfile.mkdtemp()
    try:
        iaa2, f2 = PR.choose_iaa(v1, entries, "now")
        assert iaa2 == MAN["iaa_subset"] and f2["iaa_history"][0]["subset"] == v1["iaa_subset"]
        # a done record on old-subset image 10 (a copy) is kept in the subset
        json.dump({"records": [{"id": 10, "done": True, "h_dots": [], "v_dots": []}]},
                  open(os.path.join(PR.SAVED, "annotations_x.json"), "w"))
        iaa3, f3 = PR.choose_iaa(v1, entries, "now")
        assert 10 in iaa3 and f3["iaa_history"][0]["kept_because_done"] == [10]
    finally:
        PR.SAVED = old_saved


def same(a, b):
    """equal, with None / NaN both meaning 'missing'"""
    miss = lambda x: x is None or (isinstance(x, float) and np.isnan(x))
    return (miss(a) and miss(b)) or a == b


def own_labels(ids, rng):
    rows = []
    for i in ids:
        rows.append(dict(id=i, h_tpi=float(rng.randint(20, 200)), v_tpi=float(rng.randint(20, 200)),
                         h_count=float(rng.randint(20, 200)), v_count=float(rng.randint(20, 200)),
                         weave=rng.choice(["plain", "twill_2_1", "satin"]),
                         warp_dir=rng.choice(["vertical", "horizontal", "unknown", None]),
                         h_angle_deg=round(rng.uniform(-10, 10), 2), v_angle_deg=round(rng.uniform(-10, 10), 2),
                         uncertain=rng.random() < 0.2, n_annotations=1))
    return pd.DataFrame(rows).set_index("id")


@t
def test_propagation_rules():
    """propagate: rep labels to every member; identity for k even, h<->v (+angles, warp) for k odd"""
    sc = PS.load_scenes(MAN)
    reps = sorted(sc[sc.is_scene_rep].index)
    own = own_labels(reps, random.Random(1))
    pf, ps = PS.propagate(own, sc)
    assert len(ps) == 76 and len(pf) == 150 and (ps.label_src_id == ps.index).all()
    for i, r in pf.iterrows():
        src = own.loc[r.scene_id]
        k = E[i]["scene_rot_k"]
        if k % 2:
            assert (r.h_tpi, r.v_tpi, r.h_count, r.v_count, r.h_angle_deg, r.v_angle_deg) == \
                (src.v_tpi, src.h_tpi, src.v_count, src.h_count, src.v_angle_deg, src.h_angle_deg), i
            assert same(r.warp_dir, {"vertical": "horizontal", "horizontal": "vertical"}.get(src.warp_dir, src.warp_dir))
        else:
            assert (r.h_tpi, r.v_tpi) == (src.h_tpi, src.v_tpi) and same(r.warp_dir, src.warp_dir), i
        assert r.weave == src.weave
    # the specific scene 45 = [23, 37, 45, 138, 146]: 23/37 k=3, 138 k=1 swap; 146 k=0 identity
    for i, swapped in ((23, True), (37, True), (138, True), (146, False), (45, False)):
        assert pf.at[i, "swap_hv"] == swapped
        assert pf.at[i, "h_tpi"] == (own.at[45, "v_tpi"] if swapped else own.at[45, "h_tpi"])
    # 139 (synthetic, k = 0 vs rep 64) and 132 (k = 2 vs rep 110): no swap although +-90 vs their source
    assert not pf.at[139, "swap_hv"] and not pf.at[132, "swap_hv"]
    assert int(pf.is_test_item.sum()) == 139 and not pf.loc[[131, 132, 134], "is_test_item"].any()


@t
def test_propagation_fallback_and_pool():
    """rep missing -> best labelled member rotated into the rep frame; pool averages; own_* kept"""
    sc = PS.load_scenes(MAN)
    own = pd.DataFrame([dict(id=23, h_tpi=30.0, v_tpi=60.0, weave="plain", warp_dir="vertical"),   # k = 3
                        dict(id=138, h_tpi=31.0, v_tpi=62.0, weave="plain", warp_dir="vertical"),  # synthetic k = 1
                        dict(id=0, h_tpi=47.0, v_tpi=38.0, weave="twill_2_1", warp_dir=None),      # k = 0, rep 9
                        dict(id=9, h_tpi=46.0, v_tpi=39.0, weave="twill_2_1", warp_dir=None)]).set_index("id")
    pf, ps = PS.propagate(own, sc)
    # scene 45: rep unlabelled -> 23 (real, larger crop than 37? same) beats synthetic 138
    assert ps.at[45, "label_src_id"] == 23 and ps.at[45, "label_src_kind"] == "quarter_res_copy"
    assert (ps.at[45, "h_tpi"], ps.at[45, "v_tpi"], ps.at[45, "warp_dir"]) == (60.0, 30.0, "horizontal")
    assert (pf.at[23, "h_tpi"], pf.at[23, "v_tpi"]) == (30.0, 60.0)       # back in 23's own frame
    assert (pf.at[146, "h_tpi"], pf.at[146, "v_tpi"]) == (60.0, 30.0)
    # scene 9: the rep's own label wins over the copy's
    assert ps.at[9, "label_src_id"] == 9 and pf.at[0, "h_tpi"] == 46.0 and pf.at[0, "own_h_tpi"] == 47.0
    assert bool(pf.at[0, "has_own_label"]) and bool(pf.at[9, "has_own_label"]) and not bool(pf.at[146, "has_own_label"])
    _, pool = PS.propagate(own, sc, policy="pool")
    assert abs(pool.at[45, "h_tpi"] - 61.0) < 1e-9 and abs(pool.at[45, "v_tpi"] - 30.5) < 1e-9
    assert abs(pool.at[9, "h_tpi"] - 46.5) < 1e-9
    # the rep wins even over a copy whose tool crop is LARGER than the rep's (verify-S fix):
    # 90 (byte-identical of 86), 140 (PNG of 95), 145 (PNG of 47), 146 (PNG of 45)
    big = [(c, r) for c, r in ((90, 86), (140, 95), (145, 47), (146, 45))]
    assert all(E[c]["rect_size"][0] > E[r]["rect_size"][0] for c, r in big if c != 145) and \
        E[145]["rect_size"][0] >= E[47]["rect_size"][0]
    own2 = pd.DataFrame([dict(id=r, h_tpi=40.0 + r, v_tpi=80.0 + r) for _, r in big] +
                        [dict(id=c, h_tpi=999.0, v_tpi=888.0) for c, _ in big]).set_index("id")
    pf2, ps2 = PS.propagate(own2, sc)
    for c, r in big:
        assert ps2.at[r, "label_src_id"] == r, (c, r, ps2.at[r, "label_src_id"])
        assert (ps2.at[r, "h_tpi"], ps2.at[r, "v_tpi"]) == (40.0 + r, 80.0 + r)
        assert pf2.at[c, "own_h_tpi"] == 999.0 and pf2.at[c, "h_tpi"] == 40.0 + r   # k = 0 copies
    assert (pf2.at[40, "h_tpi"], pf2.at[40, "v_tpi"]) == (80.0 + 86, 40.0 + 86)      # 40: k = 3 copy of 86


@t
def test_core_js_parity():
    """core.js copiedLabel() == propagate_scene_labels for every copy (h, v, warp_dir)"""
    if not NODE:
        print("   SKIP: node not found")
        return
    rng = random.Random(7)
    sc = PS.load_scenes(MAN)
    reps = sorted(sc[sc.is_scene_rep].index)
    recs = {}
    for r in reps:
        S = E[r]["rect_size"][0]
        nh, nv = rng.randint(10, 120), rng.randint(10, 120)
        recs[r] = dict(id=r, rect_size=S, h_line_y=S / 2, v_line_x=S / 2,
                       h_dots=[round((j + .5) * S / nh, 1) for j in range(nh)],
                       v_dots=[round((j + .5) * S / nv, 1) for j in range(nv)],
                       h_angle_deg=rng.choice([0, 0, 3.5, -7.25]), v_angle_deg=rng.choice([0, -2.0, 6.0]),
                       warp_dir=rng.choice(["vertical", "horizontal", "unknown"]), weave="plain", done=True)
    tmp = tempfile.mkdtemp()
    jp = os.path.join(tmp, "recs.json")
    json.dump({"records": list(recs.values())}, open(jp, "w"))
    core = json.dumps(os.path.join(TOOL, "core.js"))
    man = json.dumps(os.path.join(TOOL, "data", "manifest.json"))
    js = f"""
const C = require({core}); const fs = require('fs');
const M = JSON.parse(fs.readFileSync({man}, 'utf8'));
const R = {{}}; JSON.parse(fs.readFileSync({json.dumps(jp)}, 'utf8')).records.forEach(r => {{ R[r.id] = r; }});
const out = {{}};
M.images.forEach(im => {{ const c = C.copiedLabel(R[C.sceneRepId(im)], im); out[im.id] = [c.h, c.v, c.warp_dir]; }});
process.stdout.write(JSON.stringify(out));"""
    ref = json.loads(subprocess.run([NODE, "-e", js], capture_output=True, text=True, check=True).stdout)
    own = pd.DataFrame([dict(id=r["id"], h_tpi=PS.line_tpi(r, "h"), v_tpi=PS.line_tpi(r, "v"), warp_dir=r["warp_dir"])
                        for r in recs.values()]).set_index("id")
    pf, _ = PS.propagate(own, sc)
    bad = [i for i in pf.index if abs(pf.at[i, "h_tpi"] - ref[str(i)][0]) > 1e-9 or
           abs(pf.at[i, "v_tpi"] - ref[str(i)][1]) > 1e-9 or not same(pf.at[i, "warp_dir"], ref[str(i)][2])]
    assert not bad, bad[:5]


@t
def test_reader_and_cli():
    """read_own_labels: done filter, newest per annotator, mean over annotators; CLI files"""
    tmp = tempfile.mkdtemp()
    S = 500
    def rec(i, nh, nv, ts, done=True, ann="A"):
        return dict(id=i, annotator=ann, rect_size=S, h_line_y=250, v_line_x=250, h_dots=list(range(nh)),
                    v_dots=list(range(nv)), done=done, timestamp=ts, weave="plain", warp_dir="vertical")
    json.dump({"annotator": "A", "records": [rec(45, 10, 20, "2026-10-09T00:00:00Z"), rec(9, 5, 5, "x", done=False)]},
              open(os.path.join(tmp, "a1.json"), "w"))
    json.dump({"annotator": "A", "records": [rec(45, 12, 22, "2026-10-09T01:00:00Z")]}, open(os.path.join(tmp, "a2.json"), "w"))
    json.dump({"annotator": "B", "records": [rec(45, 14, 24, "2026-10-09T00:30:00Z", ann="B")]},
              open(os.path.join(tmp, "b.json"), "w"))
    paths = sorted(os.path.join(tmp, p) for p in os.listdir(tmp))
    own = PS.read_own_labels(paths, log=lambda *a: None)
    assert list(own.index) == [45] and own.at[45, "h_tpi"] == 13.0 and own.at[45, "v_tpi"] == 23.0
    assert own.at[45, "n_annotations"] == 2
    out = os.path.join(tmp, "out")
    pf, ps = PS.run(paths, out_dir=out, manifest=os.path.join(TOOL, "data", "manifest.json"), log=lambda *a: None)
    assert os.path.exists(os.path.join(out, "labels_per_file.csv")) and os.path.exists(os.path.join(out, "labels_meta.json"))
    assert int(pf.labelled.sum()) == 5 and int(ps.labelled.sum()) == 1
    assert pf.at[23, "h_tpi"] == 23.0 and pf.at[23, "warp_dir"] == "horizontal"


print(f"\nall {n} python scene tests passed")
