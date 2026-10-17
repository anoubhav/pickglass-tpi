#!/usr/bin/env python
"""scene_map.py -- per-file scene bookkeeping of the 150 crops (76 scenes) for training splits.

Reads (never writes) camera_aug/originals_provenance.json, annotation_tool/scenes_to_annotate.txt and
the 150 rectified crops annotation_tool/data/crops/<id>.jpg; writes cnn/data/scene_map.csv:

  id, scene (= representative id from scenes_to_annotate.txt), is_rep, kind, rot_k, flip, swap,
  ncc, ncc_margin, dup_of, prov_rot_k

kind        real        a real capture (JPEG) that is not a byte-identical copy of another file
            byte_dup    byte-identical to a lower-id real JPEG of the same scene (dup_of)
            png_copy    PNG decode of a JPEG of the same scene (129, 130, 133, 140, 143, 145-149)
            nb_synth    synthesised by an earlier prep notebook (97 % crop, +-90 deg, +-10 % b/c):
                        131, 132, 134-139, 141, 142, 144 -- an AUGMENTATION of its scene, never a
                        held-out sample
rot_k/flip  dihedral transform that maps this crop onto the representative's crop
            (np.rot90(np.fliplr?(crop), rot_k) ~ rep), found by NCC of low-passed thumbnails
swap        rot_k odd -> the h / v axes are exchanged relative to the representative
prov_rot_k  rot90_k of the full/quarter pair in originals_provenance.json when this file and the
            rep form such a pair (cross-check only)

    python scene_map.py            (venv2 or venv_cnn; ~10 s)
"""
from __future__ import annotations

import json
import os
import re

import cv2
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PROV = os.path.join(ROOT, "camera_aug", "originals_provenance.json")
SCENES = os.path.join(ROOT, "annotation_tool", "scenes_to_annotate.txt")
CROPS = os.path.join(ROOT, "annotation_tool", "data", "crops")
OUT = os.path.join(HERE, "data", "scene_map.csv")
NB_SYNTH = {131, 132, 134, 135, 136, 137, 138, 139, 141, 142, 144}
PNG_COPY = {129, 130, 133, 140, 143, 145, 146, 147, 148, 149}


def read_scenes(path=SCENES):
    reps = {}
    for line in open(path):
        if line.startswith("#") or not line.strip():
            continue
        r, ids = line.split(":", 1)
        reps[int(r)] = [int(x) for x in re.findall(r"\d+", ids)]
    return reps


def thumb(i, n=96, lowpass=True):
    im = cv2.imread(os.path.join(CROPS, f"{i}.jpg"), cv2.IMREAD_GRAYSCALE).astype(np.float32)
    if lowpass:   # suppress the yarn texture (periodic, ~rotation-ambiguous); keep illumination / stains
        im = cv2.GaussianBlur(im, (0, 0), im.shape[1] / 40.0)
    t = cv2.resize(im, (n, n), interpolation=cv2.INTER_AREA)
    t = t[8:-8, 8:-8]                                # ignore the border (frame-trim differences)
    return (t - t.mean()) / (t.std() + 1e-6)


def dihedral(t, k, flip):
    return np.rot90(np.fliplr(t) if flip else t, k)


def main():
    prov = json.load(open(PROV))
    reps = read_scenes()
    scene_of = {i: r for r, ids in reps.items() for i in ids}
    assert sorted(scene_of) == list(range(150)), "scenes_to_annotate.txt must cover the 150 ids once"
    groups = {frozenset(g) for g in prov["groups"]}
    assert groups == {frozenset(v) for v in reps.values()}, "scenes_to_annotate.txt != provenance groups"
    md5 = {f["id"]: f["md5"] for f in prov["files"]}
    pairs = {}
    for p in prov["fine_pairs"]:
        pairs[(p["full_res"], p["quarter_res"])] = p["rot90_k"]
    rows = []
    for i in range(150):
        r = scene_of[i]
        if i in NB_SYNTH:
            kind, dup = "nb_synth", None
        elif i in PNG_COPY:
            kind, dup = "png_copy", None
        else:
            same = sorted(j for j in reps[r] if j < i and md5[j] == md5[i] and j not in NB_SYNTH | PNG_COPY)
            kind, dup = ("byte_dup", same[0]) if same else ("real", None)
        if i == r:
            k, fl, ncc, marg = 0, False, 1.0, np.nan
        else:
            a, b = thumb(i), thumb(r)
            sc = {(k, fl): float((dihedral(a, k, fl) * b).mean()) for k in range(4) for fl in (False, True)}
            (k, fl), ncc = max(sc.items(), key=lambda kv: kv[1])
            other = max(v for (kk, ff), v in sc.items() if kk % 2 != k % 2)
            marg = ncc - other
        pk = pairs.get((r, i), pairs.get((i, r)))
        rows.append(dict(id=i, scene=r, is_rep=i == r, kind=kind, rot_k=k, flip=fl, swap=bool(k % 2),
                         ncc=round(ncc, 4), ncc_margin=round(marg, 4) if np.isfinite(marg) else np.nan,
                         dup_of=dup, prov_rot_k=pk))
    d = pd.DataFrame(rows)
    # PNG copies: which JPEG of the scene do they decode?  (fine-scale NCC at identical orientation)
    for i in sorted(PNG_COPY):
        r = scene_of[i]
        best = None
        a = thumb(i, 256, lowpass=False)
        for j in reps[r]:
            if j == i or j in NB_SYNTH | PNG_COPY:
                continue
            b = thumb(j, 256, lowpass=False)
            for k in range(4):
                for fl in (False, True):
                    s = float((dihedral(a, k, fl) * b).mean())
                    if best is None or s > best[0]:
                        best = (s, j)
        d.loc[d.id == i, "dup_of"] = best[1]
        d.loc[d.id == i, "dup_ncc"] = round(best[0], 4)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    d.to_csv(OUT, index=False)
    # ---- report
    nr = d[~d.is_rep]
    print(f"wrote {OUT}: {len(d)} files, {d.scene.nunique()} scenes; kinds {d.kind.value_counts().to_dict()}")
    print(f"non-rep files: {len(nr)}; swapped vs rep: {int(nr.swap.sum())}; min NCC {nr.ncc.min():.3f}, "
          f"min parity margin {nr.ncc_margin.min():.3f}")
    chk = nr[nr.prov_rot_k.notna()]
    if len(chk):
        # provenance rot90_k relates the two FILES; the crops inherit it up to a 180 deg ambiguity of
        # the rectification -> compare the parity (= whether h / v swap), which is what the labels need
        agree = (chk.prov_rot_k.astype(int) % 2 == chk.rot_k % 2)
        print(f"parity vs provenance rot90_k (rep pairs): {int(agree.sum())}/{len(chk)} agree")
        if not agree.all():
            print(chk[~agree].to_string())
    low = nr[(nr.ncc < 0.5) | (nr.ncc_margin < 0.1)]
    if len(low):
        print("LOW-CONFIDENCE orientation matches:")
        print(low.to_string())


if __name__ == "__main__":
    main()
