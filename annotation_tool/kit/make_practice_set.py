#!/usr/bin/env python3
"""make_practice_set.py -- (re)generate the synthetic demo / practice dataset of the annotation tool.

    python3 annotation_tool/kit/make_practice_set.py --out /tmp/practice           # the 3 fabrics in demo/data
    python3 annotation_tool/kit/make_practice_set.py --out /tmp/p --seeds 801:986514,802:969471 --max-side 1800

Each fabric is cnn/synth.py's generate(seed, max_crop_side) (generator profile v2): a rectified 1-inch crop with
exact labels. Writes <out>/data/manifest.json, data/crops/<id>.jpg (JPEG q92), data/crops_enh/<id>.jpg (CLAHE
display aid, as annotation_tool/prepare.py) and data/thumbs/<id>.jpg, plus <out>/ANSWERS.md with the true counts.
The defaults reproduce annotation_tool/demo/data (ids 801-803). These seeds were chosen from 800000-999999 for a
clear capture (no glare, shadow or motion blur, |rotation| < 2 deg, density drift < 1.2 %, >= 6.5 px per thread,
40-90 threads per inch on both axes) and then by eye. JPEG bytes can differ across OpenCV builds; the counts cannot.
Needs numpy and opencv-python-headless.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys

import cv2

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "cnn"))
import synth  # noqa: E402

DEFAULT = [(801, 986514), (802, 969471), (803, 830348)]
TOOL_WEAVE = {"plain": "plain", "twill21": "twill_2_1", "twill12": "twill_2_1", "twill31": "twill_3_1",
              "twill13": "twill_3_1", "twill22": "twill_2_2", "satin5": "satin", "sateen5": "satin"}


def enhance(crop):
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    tiles = max(2, int(round(crop.shape[0] / 96)))
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(tiles, tiles)).apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def write_jpg(path, img, q):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, q]):
        raise IOError(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", default=",".join(f"{i}:{s}" for i, s in DEFAULT), help="id:seed,id:seed,...")
    ap.add_argument("--max-side", type=int, default=1800)
    a = ap.parse_args()
    items = [tuple(int(x) for x in p.split(":")) for p in a.seeds.split(",")]
    images, rows = [], []
    for n, (qid, seed) in enumerate(items, 1):
        _, crop, lab = synth.generate(seed, max_crop_side=a.max_side)
        S = crop.shape[0]
        rel = {k: f"data/{k}/{qid}.jpg" for k in ("crops", "crops_enh", "thumbs")}
        write_jpg(os.path.join(a.out, rel["crops"]), crop, 92)
        shown = cv2.imread(os.path.join(a.out, rel["crops"]))
        write_jpg(os.path.join(a.out, rel["crops_enh"]), enhance(shown), 92)
        T = min(720, S)
        write_jpg(os.path.join(a.out, rel["thumbs"]), cv2.resize(crop, (T, T), interpolation=cv2.INTER_AREA), 85)
        images.append({"id": qid, "file": f"practice_{n}.jpg", "orig_w": S, "orig_h": S, "rect_size": [S, S],
                       "polygon": [[0.0, 0.0], [float(S), 0.0], [float(S), float(S)], [0.0, float(S)]],
                       "crop": rel["crops"], "crop_enh": rel["crops_enh"], "thumb": rel["thumbs"]})
        rows.append((qid, seed, lab["weave"], TOOL_WEAVE.get(lab["weave"], "other"), lab["colour_style"],
                     lab["h_count"], lab["v_count"], S))
        print(f"  #{qid}: seed {seed} {lab['weave']} {S}px  H {lab['h_count']:.2f}  V {lab['v_count']:.2f}")
    man = {"version": 1, "generated": dt.datetime.now().isoformat(timespec="seconds"), "n_images": len(images),
           "purpose": "practice demo: synthetic (computer-generated) fabrics as rectified 1-inch crops, blind "
                      "(no proposals); all ids in the blind subset",
           "iaa_subset": [i for i, _ in items],
           "weave_options": ["plain", "twill_2_1", "twill_3_1", "twill_2_2", "satin", "other", "unsure"],
           "images": images}
    os.makedirs(os.path.join(a.out, "data"), exist_ok=True)
    with open(os.path.join(a.out, "data", "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(man, f, separators=(",", ":"))
    L = ["# Practice fabrics: true counts", "", "| id | seed | weave (tool key) | colours | H line | V line | crop px |",
         "|---|---|---|---|---|---|---|"]
    for qid, seed, w, tw, cs, h, v, S in rows:
        L.append(f"| {qid} | {seed} | {w} ({tw}) | {cs} | {h:.2f} | {v:.2f} | {S} |")
    L += ["", "H line = vertical yarns crossing the horizontal guide line; V line = horizontal yarns. Within 1-2 "
          "threads is fine: the generator varies the spacing slightly across the window, as in real fabric."]
    with open(os.path.join(a.out, "ANSWERS.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print(f"wrote {a.out}/data (manifest + {len(images)} crops) and ANSWERS.md")


if __name__ == "__main__":
    main()
