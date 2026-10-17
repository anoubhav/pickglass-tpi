"""Verification check 3: propagate_labels.py on a fake 3-image annotation export.

Builds a tool-format JSON (schema 1, like annotation_tool/data/saved/annotations_*.json) with three
DONE images -- 0 (v1, level lines, warp vertical), 35 (v2, rotated guide lines +5 / -4 deg with dots
outside the rotated segment, warp horizontal), 44 (v2, -2.5 / +6 deg, low confidence -> uncertain) -- plus one NOT-done
record (image 2) that must be ignored.  Converts it to the tool's CSV with the tool's own
json2csv.js, gets the oracle threads/inch from core.js lineInfo() via node, runs the
propagate_labels.py CLI on the JSON and on the CSV (outputs only under verify/prop_test/), and checks
every output row against manifest.csv:
  * exactly the usable augmentations of 0, 35, 44 (flagged rows absent), nothing from image 2
  * k in {90, 270}: h<->v for tpi, counts, line angles; warp_dir vertical<->horizontal
    k in {0, 180}: unchanged
  * group == source_id, image == absolute path of crops/{aug_id}.jpg (exists), swap_hv == k in 90/270
  * JSON and CSV routes give identical labels; uncertain follows confidence == 1
  * review decisions (on a COPY of review_decisions.csv): accept / reject / k_deg_override obeyed
  * --include-not-done brings image 2 in
Run: <venv2>/bin/python test_propagate_fake3.py     exit 0 = all PASS
"""
import importlib.util
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
AUG = os.path.dirname(HERE)
ROOT = os.path.dirname(AUG)
TOOL = os.path.join(ROOT, "annotation_tool")
OUT = os.path.join(HERE, "prop_test")
os.makedirs(OUT, exist_ok=True)
PY = sys.executable
FAILS = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        FAILS.append(name)


def dots(n, S, rng, lo=0.0, hi=None):
    hi = S if hi is None else hi
    return sorted(float(round(x, 1)) for x in rng.uniform(lo + 1, hi - 1, n))


def record(i, S, nh, nv, rng, done=True, ha=0.0, va=0.0, warp="vertical", conf=3, weave="plain", extra_h=0):
    r = dict(schema=1, id=i, annotator="tester", rect_size=S, h_line_y=S / 2 - 7.3, v_line_x=S / 2 + 4.1,
             h_dots=dots(nh, S, rng) + ([1.0, S - 1.0][:extra_h]), h_orig=[0] * (nh + extra_h),
             v_dots=dots(nv, S, rng), v_orig=[0] * nv, blind_mode=False, proposals_seen=False, iaa=False,
             weave=weave, warp_dir=warp, flags={}, nonwoven=False, confidence=conf, notes="", done=done,
             time_spent_s=60, n_edits=nh + nv, tool_version="2026-10-07",
             created="2026-10-07T10:00:00.000Z", timestamp=f"2026-10-07T12:00:0{i % 10}.000Z")
    if ha or va:
        r.update(h_angle_deg=ha, v_angle_deg=va, angle_log=[])
    return r


def main():
    rng = np.random.default_rng(3)
    man_all = pd.read_csv(os.path.join(AUG, "manifest.csv"))
    S = {i: int(man_all[man_all.source_id == i].source_rect_size.iloc[0]) for i in (0, 35, 44, 2)}
    recs = [record(0, S[0], 61, 44, rng, warp="vertical", weave="plain"),
            record(35, S[35], 52, 37, rng, ha=5.0, va=-4.0, warp="horizontal", weave="twill_2_1", extra_h=2),
            record(44, S[44], 28, 71, rng, conf=1, ha=-2.5, va=6.0, warp="vertical", weave="satin"),
            record(2, S[2], 10, 10, rng, done=False)]
    exp = dict(schema=1, tool="wvaq2027-thread-count-annotator", exported="2026-10-07T12:00:00.000Z",
               annotator="tester", n_records=4, n_done=3, records=recs)
    jpath = os.path.join(OUT, "annotations_fake3.json")
    json.dump(exp, open(jpath, "w"), indent=1)

    # tool's own CSV export + oracle TPI from core.js
    subprocess.run(["node", os.path.join(TOOL, "json2csv.js"), jpath, "--out-dir", OUT], check=True,
                   capture_output=True)
    cpath = os.path.join(OUT, "annotations_fake3.csv")
    js = ("const C=require(%r);const o=require(%r);const out={};for(const r of o.records){"
          "out[r.id]={h:C.lineInfo(r,'h').tpi,v:C.lineInfo(r,'v').tpi,ha:C.lineInfo(r,'h').angle,"
          "va:C.lineInfo(r,'v').angle}};console.log(JSON.stringify(out))") % (os.path.join(TOOL, "core.js"), jpath)
    oracle = {int(k): v for k, v in json.loads(subprocess.run(["node", "-e", js], check=True, capture_output=True,
                                                             text=True).stdout).items()}
    print("oracle tpi (core.js):", {k: (round(v["h"], 3), round(v["v"], 3)) for k, v in oracle.items()})
    check("oracle: rotated line on 35 excludes out-of-segment dots", oracle[35]["h"] != 54)
    # NOTE: line-angle signs under a 90-degree turn are verified physically in test_swap_synthetic T8;
    # here they are checked against core.js lineInfo().angle with the swap rule

    # CLI runs (never the default output path). Pinned to the per-source behaviour this check was written
    # for (the CLI default has since become scene-aware -- labels copied to every file of a scene,
    # group = scene_id; that is tested in tests/test_propagate.py test 5)
    for src, name in ((jpath, "labels_json.csv"), (cpath, "labels_csv.csv")):
        subprocess.run([PY, os.path.join(AUG, "propagate_labels.py"), src, "--out", os.path.join(OUT, name),
                        "--no-scenes", "--group-by", "source"], check=True)
    lj = pd.read_csv(os.path.join(OUT, "labels_json.csv"))
    lc = pd.read_csv(os.path.join(OUT, "labels_csv.csv"))

    man = man_all[man_all.source_id.isin([0, 35, 44])]
    exp_ids = set(man[man.usable].aug_id)
    check("row set == usable augs of 0/35/44", set(lj.aug_id) == exp_ids,
          f"({len(lj)} rows; expected {len(exp_ids)}; flagged excluded: "
          f"{sorted(set(man[~man.usable].aug_id))})")
    check("not-done image 2 ignored", not (lj.source_id == 2).any())

    m = lj.merge(man_all[["aug_id", "k_deg"]].rename(columns={"k_deg": "k_man"}), on="aug_id")
    ok_swap = ok_meta = True
    bad = []
    for r in m.itertuples():
        o = oracle[r.source_id]
        src_rec = next(x for x in recs if x["id"] == r.source_id)
        sw = r.k_man in (90, 270)
        eh, ev = (o["v"], o["h"]) if sw else (o["h"], o["v"])
        ehc, evc = (len(src_rec["v_dots"]), len(src_rec["h_dots"])) if sw else (len(src_rec["h_dots"]), len(src_rec["v_dots"]))
        eha, eva = (o["va"], o["ha"]) if sw else (o["ha"], o["va"])
        ewd = {"vertical": "horizontal", "horizontal": "vertical"}[src_rec["warp_dir"]] if sw else src_rec["warp_dir"]
        ok = (abs(r.h_tpi - eh) < 1e-6 and abs(r.v_tpi - ev) < 1e-6 and r.h_count == ehc and r.v_count == evc
              and abs(r.h_angle_deg - eha) < 1e-9 and abs(r.v_angle_deg - eva) < 1e-9 and r.warp_dir == ewd
              and bool(r.swap_hv) == sw and r.k_deg == r.k_man and r.weave == src_rec["weave"])
        okm = (r.group == r.source_id and os.path.isabs(r.image) and os.path.exists(r.image)
               and os.path.basename(r.image) == f"{r.aug_id}.jpg"
               and bool(r.uncertain) == (src_rec["confidence"] == 1))
        ok_swap &= ok
        ok_meta &= okm
        if not (ok and okm):
            bad.append(r.aug_id)
    print(m[["aug_id", "source_id", "group", "k_deg", "swap_hv", "h_tpi", "v_tpi", "h_count", "v_count",
             "h_angle_deg", "v_angle_deg", "warp_dir", "uncertain"]].to_string(index=False))
    check("tpi / counts / angles / warp_dir swapped iff k in {90,270}, else unchanged (vs core.js oracle)", ok_swap,
          f"bad={bad}")
    check("group == source_id, image = existing abs crop path, uncertain from confidence==1", ok_meta)
    print("k distribution in test rows:", m.k_man.value_counts().sort_index().to_dict())
    check("test covers all four k", set(m.k_man) == {0, 90, 180, 270})

    cols = ["aug_id", "h_tpi", "v_tpi", "h_count", "v_count", "h_angle_deg", "v_angle_deg", "warp_dir", "weave",
            "group", "uncertain"]
    a = lj[cols].sort_values("aug_id").reset_index(drop=True)
    b = lc[cols].sort_values("aug_id").reset_index(drop=True)
    num = ["h_tpi", "v_tpi", "h_count", "v_count", "h_angle_deg", "v_angle_deg"]
    same = (a.aug_id.equals(b.aug_id) and np.allclose(a[num].values, b[num].values, atol=0.0051)   # tool CSV rounds tpi to 2 dp
            and a[["warp_dir", "weave", "group", "uncertain"]].astype(str).equals(
                b[["warp_dir", "weave", "group", "uncertain"]].astype(str)))
    check("JSON route == CSV route (tool json2csv export)", same)

    # review decisions on a COPY (the real review_decisions.csv is never touched)
    spec = importlib.util.spec_from_file_location("pl", os.path.join(AUG, "propagate_labels.py"))
    PL = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(PL)
    dec = pd.read_csv(PL.DECISIONS, dtype=str, keep_default_na=False)
    dec.loc[dec.aug_id == "35_aug4", "decision"] = "accept"
    dec.loc[dec.aug_id == "35_aug5", ["decision", "k_deg_override"]] = ["accept", "90"]
    dec = pd.concat([dec, pd.DataFrame([{c: "" for c in dec.columns} | {"aug_id": "0_aug1", "decision": "reject"}])])
    dpath = os.path.join(OUT, "review_decisions_copy.csv")
    dec.to_csv(dpath, index=False)
    PL.DECISIONS = dpath
    src, _ = PL.read_annotations([jpath], log=lambda *x: None)
    lab = PL.propagate(src, PL.load_manifest()).set_index("aug_id")
    o35 = oracle[35]
    ok_dec = ("35_aug4" in lab.index and "35_aug5" in lab.index and "0_aug1" not in lab.index
              and abs(lab.loc["35_aug4", "h_tpi"] - o35["v"]) < 1e-6                  # k 270 -> swapped
              and int(lab.loc["35_aug5", "k_deg"]) == 90 and bool(lab.loc["35_aug5", "swap_hv"])
              and abs(lab.loc["35_aug5", "h_tpi"] - o35["v"]) < 1e-6)
    check("review decisions obeyed (accept, reject, k_deg_override) on a copy", ok_dec)

    src2, _ = PL.read_annotations([jpath], include_not_done=True, log=lambda *x: None)
    check("--include-not-done brings image 2 in", 2 in set(src2.source_id))
    print("ALL PASS" if not FAILS else f"FAILED: {FAILS}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
