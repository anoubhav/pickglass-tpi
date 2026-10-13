#!/usr/bin/env python
"""Scene structure of the 150 dataset files (which files are the same photograph).

The 150 files in fabric_notebooks/final_image_set/ are only 76 distinct scenes
(camera_aug/originals_provenance.json, camera_aug/RESEARCH.md section 1):
  * 5 groups of byte-identical files,
  * 62 full-resolution / quarter-resolution pairs of the SAME exposure (often rotated 90 deg),
  * 10 PNG files that are decodes of JPEGs in the set,
  * 11 PNG files synthesised by the 2024 preparation notebook (97 % centre crop, a +-90 deg
    rotation, +-10 % brightness / contrast) -- augmentations, not captures.

This module turns that into one record per file (used by prepare.py for the tool manifest,
by propagate_scene_labels.py and by the scoring / CV scripts):

  scene_id        id of the scene's representative file (the photo that is annotated)
  is_scene_rep    True for exactly one file per scene: the highest-resolution real JPEG
                  (ties: lowest id); must agree with scenes_to_annotate.txt when present
  scene_members   sorted ids of all files of the scene
  kind            rep | byte_identical | quarter_res_copy | png_copy | notebook_synthetic
                    byte_identical      JPEG with the same md5 as the rep or a lower-id member
                    quarter_res_copy    smaller JPEG of the same exposure (4x reduced)
                    png_copy            PNG with exactly the size of a JPEG member (decoded copy)
                    notebook_synthetic  PNG of 0.97 x a JPEG member's size (notebook augmentation)
  copy_of         byte_identical / png_copy: the file it duplicates; notebook_synthetic: the
                  JPEG it was made from (derived_from); None otherwise
  scene_rot_k     k in 0..3 with  crop(file) ~= numpy.rot90(crop(rep), k)  (counter-clockwise,
                  the aug1016 convention); k odd -> the file's H line crosses the yarns that the
                  rep's V line crosses, so h <-> v (and warp_dir) swap when labels are copied
  scene_rot_ncc / scene_rot_margin   best NCC and best minus second-best NCC of the 4 hypotheses

Rotation recovery (as aug1016/aug_common.py does for the 2024 augmentations): the rep is
re-rectified with its canonical corners rolled by j (j = 0..3), which puts every rotation
hypothesis on exactly the pixel grid of a rectify.py crop; features are CLAHE grey and gradient
magnitude, low-passed (sigma 2 px at 256 px/inch) and compared at 128 px/inch over the inner
84 % against the file's own tool crop (data/crops/<id>.jpg, i.e. what the annotator sees).

    <venv2>/bin/python scenes.py                 # print the table + margins (crop NCC)
    <venv2>/bin/python scenes.py --check-images  # + independent whole-photo NCC check -> data/scenes_check.csv

Read-only for everything outside annotation_tool/data/ (the check CSV is the only file written).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)  # .../wvaq2027
sys.path.insert(0, os.path.join(ROOT, "common"))
import rectify as R  # noqa: E402

PROVENANCE = os.path.join(ROOT, "camera_aug", "originals_provenance.json")
REPS_FILE = os.path.join(HERE, "scenes_to_annotate.txt")
CROPS = os.path.join(HERE, "data", "crops")
CHECK_CSV = os.path.join(HERE, "data", "scenes_check.csv")

KINDS = ("rep", "byte_identical", "quarter_res_copy", "png_copy", "notebook_synthetic")
SYNTH_CROP = 0.97          # notebook cell 18: 97 % centre crop
FEAT_N = 128               # NCC working resolution (per 1-inch crop side), as aug1016
FEAT_MARGIN = 0.08
FEAT_SIGMA = 2.0
_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


# --------------------------------------------------------------------------- structure
def load_provenance(path=PROVENANCE):
    with open(path) as f:
        d = json.load(f)
    files = {int(e["id"]): e for e in d["files"]}
    groups = [sorted(int(i) for i in g) for g in d["groups"]]
    return files, groups


def read_reps_file(path=REPS_FILE):
    """{rep_id: [member ids]} from scenes_to_annotate.txt ('rep: [a, b, ...]' lines), or None."""
    if not path or not os.path.exists(path):
        return None
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or ":" not in line:
                continue
            k, v = line.split(":", 1)
            out[int(k)] = sorted(int(x) for x in v.strip().strip("[]").split(",") if x.strip())
    return out


def _px(e):
    return int(e["size"][0]) * int(e["size"][1])


def _same_size(a, b):
    return sorted(a) == sorted(b)


def _synth_of(png_size, jpg_size):
    """PNG size == round-ish(0.97 x JPEG size) in either orientation (the notebook crop)."""
    w, h = sorted(png_size)
    W, H = sorted(jpg_size)
    return abs(w - SYNTH_CROP * W) <= 1.5 and abs(h - SYNTH_CROP * H) <= 1.5


def scene_structure(files=None, groups=None, reps_file=REPS_FILE):
    """{id: {scene_id, is_scene_rep, scene_members, kind, copy_of}} for all files (no NCC)."""
    if files is None:
        files, groups = load_provenance()
    out = {}
    for g in groups:
        jpgs = [i for i in g if files[i]["container"] == "JPEG"]
        if not jpgs:
            raise ValueError(f"scene {g} has no real JPEG")
        rep = sorted(jpgs, key=lambda i: (-_px(files[i]), i))[0]
        first_md5 = {}
        for i in sorted(g):
            first_md5.setdefault(files[i]["md5"], i)
        for i in g:
            e = files[i]
            copy_of = None
            if i == rep:
                kind = "rep"
            elif e["container"] == "PNG":
                same = [j for j in sorted(jpgs, key=lambda j: (j != rep, j)) if _same_size(e["size"], files[j]["size"])]
                syn = [j for j in sorted(jpgs, key=lambda j: (j != rep, j)) if _synth_of(e["size"], files[j]["size"])]
                if same:
                    kind, copy_of = "png_copy", same[0]
                elif syn:
                    kind, copy_of = "notebook_synthetic", syn[0]
                else:
                    raise ValueError(f"file {i}: PNG of unexplained size {e['size']}")
            elif e["md5"] == files[rep]["md5"]:
                kind, copy_of = "byte_identical", rep
            elif first_md5[e["md5"]] != i:
                kind, copy_of = "byte_identical", first_md5[e["md5"]]
            elif _px(e) < _px(files[rep]):
                kind = "quarter_res_copy"
            else:
                raise ValueError(f"file {i}: same-size JPEG of scene {rep} but not byte-identical")
            out[i] = dict(scene_id=rep, is_scene_rep=i == rep, scene_members=sorted(g), kind=kind, copy_of=copy_of)
    reps = read_reps_file(reps_file)
    if reps is not None:
        mine = {s["scene_id"]: s["scene_members"] for s in out.values()}
        if mine != reps:
            diff = sorted(k for k in set(mine) | set(reps) if mine.get(k) != reps.get(k))[:6]
            raise ValueError(f"{reps_file} disagrees with the provenance groups / rep rule (scenes {diff})")
    return out


# --------------------------------------------------------------------------- rotation (crop NCC)
def ncc_features(crop_bgr, sigma=FEAT_SIGMA):
    """Same features as aug1016/aug_common.ncc_features (CLAHE grey + gradient magnitude)."""
    g = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (256, 256), interpolation=cv2.INTER_AREA if g.shape[0] > 256 else cv2.INTER_CUBIC)
    c = cv2.GaussianBlur(_CLAHE.apply(g).astype(np.float32), (0, 0), sigma)
    gs = cv2.GaussianBlur(g.astype(np.float32), (0, 0), 1.5)
    gm = cv2.GaussianBlur(cv2.magnitude(cv2.Sobel(gs, cv2.CV_32F, 1, 0), cv2.Sobel(gs, cv2.CV_32F, 0, 1)),
                          (0, 0), sigma)
    return [cv2.resize(f, (FEAT_N, FEAT_N), interpolation=cv2.INTER_AREA) for f in (c, gm)]


def _std(f, m):
    f = f[m:FEAT_N - m, m:FEAT_N - m].astype(np.float64)
    return (f - f.mean()) / (f.std() + 1e-9)


def rep_rotations(rep_id, img=None):
    """rot90^j (CCW, j = 0..3) of the rep's crop, re-rectified with the corners rolled by j."""
    img = R.load_image(rep_id) if img is None else img
    corners = R.load_corners(rep_id)
    out = []
    for j in range(4):
        H, S = R.homography(None, corners=np.roll(np.asarray(corners), -j, axis=0))
        out.append(cv2.warpPerspective(img, H, (S, S), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE))
    return out


def ncc_k(rep_feats_by_j, file_feats):
    """-> (k, best, margin, scores[4]) with scores[j] = mean over the 2 features of the NCC."""
    m = int(round(FEAT_N * FEAT_MARGIN))
    a = [_std(f, m) for f in file_feats]
    sc = np.array([np.mean([np.mean(_std(rep_feats_by_j[j][q], m) * a[q]) for q in range(2)]) for j in range(4)])
    order = np.argsort(-sc)
    return int(order[0]), float(sc[order[0]]), float(sc[order[0]] - sc[order[1]]), sc


def tool_crop(i, crops_dir=CROPS):
    p = os.path.join(crops_dir, f"{i}.jpg")
    c = cv2.imread(p)
    if c is None:
        c = R.rectify(i)
    return c


def compute_scenes(crops_dir=CROPS, reps_file=REPS_FILE, log=print):
    """Full per-file scene table incl. scene_rot_k (crop NCC). Deterministic."""
    files, groups = load_provenance()
    st = scene_structure(files, groups, reps_file)
    for rep in sorted({s["scene_id"] for s in st.values()}):
        members = st[rep]["scene_members"]
        rot = rep_rotations(rep)
        feats = [ncc_features(c) for c in rot]
        for i in members:
            if i == rep:
                st[i].update(scene_rot_k=0, scene_rot_ncc=1.0, scene_rot_margin=None)
                continue
            k, best, margin, sc = ncc_k(feats, ncc_features(tool_crop(i, crops_dir)))
            st[i].update(scene_rot_k=k, scene_rot_ncc=round(best, 4), scene_rot_margin=round(margin, 4),
                         _scores=[round(float(x), 4) for x in sc])
    # rotation relative to the file a copy / synthetic file was made from (k_rel = k_file - k_src)
    for i, s in st.items():
        src = s["copy_of"]
        s["rot_k_vs_copy_of"] = None if src is None else int((s["scene_rot_k"] - st[src]["scene_rot_k"]) % 4)
    return st


# --------------------------------------------------------------------------- independent check
def _thumb(img, long_side=192):
    s = long_side / max(img.shape[:2])
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.resize(g, (max(1, round(img.shape[1] * s)), max(1, round(img.shape[0] * s))), interpolation=cv2.INTER_AREA)


def image_level_k(rep_img, file_img, synthetic):
    """Whole-photo check, independent of the window polygons: NCC of low-passed grey thumbnails
    of the file against rot90^j of the rep photo (centre-cropped to 97 % for notebook files).
    Hypotheses whose aspect ratio does not match the file are skipped (score nan)."""
    fh, fw = file_img.shape[:2]
    ft = cv2.GaussianBlur(_thumb(file_img).astype(np.float32), (0, 0), 1.5)
    sc = np.full(4, np.nan)
    for j in range(4):
        r = np.ascontiguousarray(np.rot90(rep_img, j))
        if synthetic:
            H, W = r.shape[:2]
            h, w = int(round(H * SYNTH_CROP)), int(round(W * SYNTH_CROP))
            r = r[(H - h) // 2:(H - h) // 2 + h, (W - w) // 2:(W - w) // 2 + w]
        if abs(r.shape[1] / r.shape[0] - fw / fh) > 0.02:
            continue
        rt = cv2.GaussianBlur(cv2.resize(_thumb(r), (ft.shape[1], ft.shape[0]), interpolation=cv2.INTER_AREA)
                              .astype(np.float32), (0, 0), 1.5)
        a, b = (ft - ft.mean()) / (ft.std() + 1e-9), (rt - rt.mean()) / (rt.std() + 1e-9)
        sc[j] = float(np.mean(a * b))
    valid = np.where(np.isfinite(sc))[0]
    if not len(valid):
        return None, None, None, sc
    order = valid[np.argsort(-sc[valid])]
    margin = float(sc[order[0]] - sc[order[1]]) if len(order) > 1 else None
    return int(order[0]), float(sc[order[0]]), margin, sc


def check_images(st, out_csv=CHECK_CSV, log=print):
    import pandas as pd
    rows = []
    cache = {}
    for i in sorted(st):
        s = st[i]
        if s["is_scene_rep"]:
            continue
        rep = s["scene_id"]
        if rep not in cache:
            cache.clear()
            cache[rep] = R.load_image(rep)
        k, best, margin, sc = image_level_k(cache[rep], R.load_image(i), s["kind"] == "notebook_synthetic")
        rows.append(dict(id=i, scene_id=rep, kind=s["kind"], copy_of=s["copy_of"], k_crop=s["scene_rot_k"],
                         ncc_crop=s["scene_rot_ncc"], margin_crop=s["scene_rot_margin"],
                         ncc_crop_j=" ".join(f"{x:.3f}" for x in s["_scores"]),
                         k_image=k, ncc_image=None if best is None else round(best, 4),
                         margin_image=None if margin is None else round(margin, 4),
                         ncc_image_j=" ".join("nan" if not np.isfinite(x) else f"{x:.3f}" for x in sc),
                         agree=k == s["scene_rot_k"]))
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    log(f"image-level check: k agrees for {int(df.agree.sum())}/{len(df)} non-rep files -> {out_csv}")
    return df


def md5_of(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def summarise(st, log=print):
    from collections import Counter
    reps = sorted({s["scene_id"] for s in st.values()})
    log(f"{len(st)} files, {len(reps)} scenes; kinds {dict(Counter(s['kind'] for s in st.values()))}")
    nr = [s for s in st.values() if not s["is_scene_rep"]]
    for kind in KINDS[1:]:
        ks = [s for s in nr if s["kind"] == kind]
        if not ks:
            continue
        mg = np.array([s["scene_rot_margin"] for s in ks])
        nc = np.array([s["scene_rot_ncc"] for s in ks])
        log(f"  {kind:19s} n={len(ks):2d}  k(0/1/2/3)={[sum(1 for s in ks if s['scene_rot_k'] == k) for k in range(4)]}"
            f"  NCC best min {nc.min():.3f} median {np.median(nc):.3f}  margin min {mg.min():.3f} median {np.median(mg):.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check-images", action="store_true", help="also run the whole-photo NCC check (~1 min)")
    ap.add_argument("--json", help="write the per-file table to this JSON file")
    a = ap.parse_args()
    st = compute_scenes()
    summarise(st)
    for i in sorted(st):
        s = st[i]
        if not s["is_scene_rep"]:
            print(f"  {i:3d} scene {s['scene_id']:3d} {s['kind']:19s} copy_of={s['copy_of']!s:4s} "
                  f"k={s['scene_rot_k']} k_vs_src={s['rot_k_vs_copy_of']!s:4s} ncc={s['scene_rot_ncc']:.3f} "
                  f"margin={s['scene_rot_margin']:.3f}  j-scores {s['_scores']}")
    if a.check_images:
        check_images(st)
    if a.json:
        with open(a.json, "w") as f:
            json.dump({str(k): {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in sorted(st.items())},
                      f, indent=1)


if __name__ == "__main__":
    main()
