#!/usr/bin/env python3
"""count_photos.py -- count threads on NEW pick-glass photos with the released pipeline (no labels needed).

    python3 tools/count_photos.py photos/*.jpg --out results.csv --crops-dir crops/
    python3 tools/count_photos.py photos/*.jpg --corners my_corners.csv --out results.csv
    python3 tools/count_photos.py photos/*.jpg --first-id 150 --polygons-out new_polygons.csv

Per photo:
  1. window corners: from --corners (CSV with columns file,x1,y1,x2,y2,x3,y3,x4,y4 in original pixels, any order)
     or, by default, from the glass's engraved scales (methods/ruler_calib.py calibrate(); tuned for the pick-glass
     model used for PickGlass: 26 fine ticks + 11 coarse ticks; for another glass give the corners by hand);
  2. rectification to an S x S crop of the opening, S = mean side length in px (common/rectify.py convention);
  3. the FROZEN counter (methods/results/counting_frozen_final.py with methods/conf_model.json): h, v, weave,
     confidence, and the dot positions on the two centre guide lines;
  4. capture checks: px per yarn on the denser axis (< 4: counts break; 4-7: lattice flips possible; >= 7: safe),
     ruler quality, counter confidence.
Writes one CSV row per photo (and optionally the crops, and window-polygon rows in the data/window_polygons.csv
format so the photos can be added to an annotation set; see docs/how_to_extend.md).

Counts are threads per WINDOW SIDE; they equal threads per inch only if the opening is 1 inch at the fabric plane
(the glass used for PickGlass measures about 26.0 fine divisions on its own scale: calibrate your glass, see
docs/open_problems.md). A screening estimate, never a certification (MODEL_CARD.md).
Needs numpy, scipy, opencv-python-headless.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys

import cv2
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "common"))
sys.path.insert(0, os.path.join(REPO, "methods"))
import rectify as R  # noqa: E402
import ruler_calib as RC  # noqa: E402


def frozen_counter():
    spec = importlib.util.spec_from_file_location(
        "counting_frozen_final", os.path.join(REPO, "methods", "results", "counting_frozen_final.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m.CONF_MODEL = json.load(open(os.path.join(REPO, "methods", "conf_model.json")))
    return m


def read_corners(path):
    out = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            out[os.path.basename(r["file"])] = R.order_corners(
                [[float(r[f"x{k}"]), float(r[f"y{k}"])] for k in range(1, 5)])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("photos", nargs="+")
    ap.add_argument("--corners", help="CSV file,x1..y4 (skips the ruler)")
    ap.add_argument("--out", default="count_photos.csv")
    ap.add_argument("--crops-dir", help="also write the rectified crops (<stem>.jpg, JPEG q92)")
    ap.add_argument("--polygons-out", help="write window polygons in the data/window_polygons.csv layout")
    ap.add_argument("--first-id", type=int, default=150, help="first id for --polygons-out (150 = after the 150 PickGlass files)")
    a = ap.parse_args()
    C = frozen_counter()
    given = read_corners(a.corners) if a.corners else {}
    rows, polys = [], []
    for n, p in enumerate(a.photos):
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        name = os.path.basename(p)
        row = dict(file=name, ok=False)
        if img is None:
            row["note"] = "unreadable"
            rows.append(row)
            continue
        if name in given:
            corners, src, q = given[name], "given", None
        else:
            cal = RC.calibrate(img)
            if not cal.get("ok"):
                row["note"] = f"ruler not found ({cal.get('fail')}): give the corners with --corners"
                rows.append(row)
                print(f"{name}: {row['note']}")
                continue
            corners, src, q = R.order_corners(cal["corners"]), "ruler", cal["quality"]["score"]
        Hm, S = R.homography(None, corners=corners)
        crop = cv2.warpPerspective(img, Hm, (S, S), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        est = C.estimate(crop, method="ours")
        h, v = float(est["h_count"]), float(est["v_count"])
        dens = max(x for x in (h, v) if np.isfinite(x)) if any(np.isfinite([h, v])) else float("nan")
        ppy = S / dens if dens and np.isfinite(dens) else float("nan")
        warn = []
        if np.isfinite(ppy) and ppy < 4:
            warn.append("fewer than 4 px per yarn: count unreliable, retake closer / at full resolution")
        elif np.isfinite(ppy) and ppy < 7:
            warn.append("4-7 px per yarn: lattice flips possible")
        if est["confidence"] < 0.5:
            warn.append("low confidence: check the dots or retake")
        if src == "ruler" and q is not None and q < 0.3:
            warn.append("weak ruler fit: check the corners")
        row.update(ok=True, corners_from=src, ruler_quality=None if q is None else round(q, 3), S_px=S,
                   h_per_window=round(h, 2), v_per_window=round(v, 2), weave=est["weave"],
                   confidence=round(float(est["confidence"]), 3), px_per_yarn=round(ppy, 2),
                   note="; ".join(warn))
        rows.append(row)
        print(f"{name}: H {h:.1f}  V {v:.1f}  {est['weave']}  conf {est['confidence']:.2f}  "
              f"{ppy:.1f} px/yarn  ({src}){'  ! ' + row['note'] if warn else ''}")
        if a.crops_dir:
            os.makedirs(a.crops_dir, exist_ok=True)
            cv2.imwrite(os.path.join(a.crops_dir, os.path.splitext(name)[0] + ".jpg"), crop,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
        c = np.asarray(corners, float)
        polys.append([a.first_id + n, name, img.shape[1], img.shape[0]] + [repr(float(x)) for x in c.reshape(-1)] +
                     [repr(float(x)) for x in c.reshape(-1)] + [S, ""])
    keys = ["file", "ok", "corners_from", "ruler_quality", "S_px", "h_per_window", "v_per_window", "weave",
            "confidence", "px_per_yarn", "note"]
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})
    print(f"wrote {a.out} ({sum(r['ok'] for r in rows)}/{len(rows)} counted)")
    if a.polygons_out:
        with open(a.polygons_out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "file", "orig_w", "orig_h", "x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4",
                        "tl_x", "tl_y", "tr_x", "tr_y", "br_x", "br_y", "bl_x", "bl_y", "side_px",
                        "manifest_polygon_match"])
            w.writerows(polys)
        print(f"wrote {a.polygons_out} ({len(polys)} polygons, ids from {a.first_id})")


if __name__ == "__main__":
    main()
