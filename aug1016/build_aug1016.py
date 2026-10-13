#!/usr/bin/env python
"""build_aug1016.py -- rectify the 2024 augmented set and recover each augmentation's
unrecorded 90-degree rotation.

Inputs (read-only):
  $PICKGLASS_NOTEBOOKS/final_image_set/{i}.jpeg + final_bbox_annotation_set/{i}.xml
  $PICKGLASS_NOTEBOOKS/augmented_images/{i}_aug{n}.jpeg + augmented_xml/{i}_aug{n}.xml
Outputs (this folder):
  crops/{i}_aug{n}.jpg       rectified 1-inch crop of every augmentation on disk, made with
                             common/rectify.py's convention (JPEG q=92 like the 150 tool crops)
  manifest.csv               one row per augmentation (columns: README.md)
  missing_augmentations.csv  the 47 of 900 generated augmentations not on disk, and why
  review/contact_flagged.png    every row that is not auto-accepted
  review/contact_ok_sample.png  random sample of accepted rows (sanity)
  review/review_decisions.csv   flagged rows + a suggested decision; fill `decision`
                                (accept / reject) to override, propagate_labels.py obeys it

    <venv2>/bin/python build_aug1016.py [--workers 6] [--no-crops]

k recovery, two independent ways:
  (a) geometric : where each STORED corner (point1..4) lands about the polygon centroid in the
                  augmentation vs in the original (albumentations keeps key-point order).
  (b) photometric: NCC between the aug crop and the source crop rotated by j*90 deg (CCW),
                  j = 0..3, on low-passed CLAHE grey + gradient magnitude at 128 px/inch;
                  margin = best - second best.
"""
from __future__ import annotations

import os as _os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_k, "1")

import argparse
import json
import os
import re
import time
from multiprocessing import Pool

import cv2

cv2.setNumThreads(1)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import aug_common as A  # noqa: E402

R = A.R
NOTEBOOK = os.path.join(A.NB_DIR, "1_Segmentation_Dataset.ipynb")
REVIEW_DIR = os.path.join(A.HERE, "review")

# acceptance thresholds (chosen from the distributions, see README "Thresholds")
NCC_MIN = 0.5          # best mean-NCC
MARGIN_MIN = 0.15      # best - second-best mean-NCC (fabric 22 is 180-degree symmetric: 0.18-0.27)
GEO_RESID_MAX = 25.0   # per-corner angular inconsistency of the geometric estimate (deg)
# transform-presence thresholds (each is a clear valley of a bimodal histogram, README)
PERSP_PERMILLE = 0.4   # affine-fit residual / image diagonal  (x 1000)
BC_GAIN_TOL, BC_OFFSET_TOL = 0.02, 2.0
NOISE_RATIO = 1.5      # Immerkaer sigma(aug) / sigma(source)
BLOCK_SMALL, BLOCK_BIG = 1.13, 1.17   # JPEG 8x8 blockiness, frames < / >= 2000 px wide


def fmt_poly(p):
    return " ".join(f"{v:.0f}" if float(v).is_integer() else f"{v:.2f}" for v in np.asarray(p, float).ravel())


def process_source(args):
    i, ns, write_crops = args
    img = R.load_image(i)
    src_pts, src_size = A.read_voc_polygon(R.annotation_path(i))
    src, src_canon, S_src = A.rectify_canonical(img, src_pts)        # == rectify.rectify(i)
    src_rots = A.source_rotations(img, src_canon, S_src)              # rot90^j hypotheses
    feats_rots = [A.ncc_features(c)[0] for c in src_rots]
    m = A.ncc_features(src)[1]
    # stored-order convention (point1 -> top-left): only used as evidence, see README
    feats_st = [A.ncc_features(c)[0] for c in A.source_rotations(img, src_pts, S_src)]
    noise_src = A.noise_sigma(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    rows = []
    for n in ns:
        t0 = time.time()
        aid = A.aug_id(i, n)
        aimg = cv2.imread(A.aug_image_path(i, n), cv2.IMREAD_COLOR)
        agrey = cv2.cvtColor(aimg, cv2.COLOR_BGR2GRAY)
        ap, asize = A.read_voc_polygon(A.aug_xml_path(i, n))
        probs = A.polygon_problems(ap, asize, aimg.shape)
        stale = A.stale_points(src_pts, ap)
        if stale:
            probs.append("stale_keypoints")
        crop, canon, S = A.rectify_canonical(aimg, ap)
        if write_crops:
            os.makedirs(A.CROPS_DIR, exist_ok=True)
            tmp = A.crop_path(i, n) + ".tmp.jpg"
            cv2.imwrite(tmp, crop, [cv2.IMWRITE_JPEG_QUALITY, A.JPEG_Q])
            os.replace(tmp, A.crop_path(i, n))
        # (a) geometric
        g = A.k_geometric(src_pts, ap)
        # (b) photometric
        fa, _ = A.ncc_features(crop)
        nc = A.ncc_rot(feats_rots, fa, m)
        order = np.argsort(-nc[:, 2])
        j_best, j_2nd = int(order[0]), int(order[1])
        # evidence for the corner-convention question: stored-order rectification
        ncs = A.ncc_rot(feats_st, A.ncc_features(A.rectify_stored_order(aimg, ap))[0], m)
        k_photo = 90 * j_best
        rt = A.residual_transform(src_pts, src_size, ap, k_photo)
        pf = A.photometric_fit(src_rots[j_best], crop)
        row = dict(
            aug_id=aid, source_id=i, aug_index=n,
            k_deg=k_photo, swap_hv=k_photo in (90, 270),
            k_geo_deg=g["k_deg"], k_photo_deg=k_photo, agree=g["k_deg"] == k_photo,
            geo_resid_deg=round(g["resid_deg"], 2), geo_slot_src=g["idx_src"], geo_slot_aug=g["idx_aug"],
            ncc_best=round(nc[j_best, 2], 4), ncc_second=round(nc[j_2nd, 2], 4),
            ncc_margin=round(nc[j_best, 2] - nc[j_2nd, 2], 4),
            ncc_clahe_best=round(nc[j_best, 0], 4), ncc_grad_best=round(nc[j_best, 1], 4),
            **{f"ncc_j{90 * j}": round(nc[j, 2], 4) for j in range(4)},
            stored_order_best_deg=90 * int(np.argmax(ncs[:, 2])), stored_order_ncc0=round(ncs[0, 2], 4),
            polygon=fmt_poly(ap), polygon_canonical=fmt_poly(canon), stale_points=" ".join(map(str, stale)),
            img_w=asize[0], img_h=asize[1], rect_size=S, source_rect_size=S_src,
            rect_ratio=round(S / S_src, 4),
            area_pct=round(100 * A.shoelace(ap) / (asize[0] * asize[1]), 2),
            persp_rms_px=round(rt["persp_rms_px"], 2), affine_scale=round(rt["affine_scale"], 4),
            aniso=round(rt["aniso"], 4), shift_px=round(rt["shift_px"], 1),
            bc_gain=round(pf["gain"], 4), bc_offset=round(pf["offset"], 2), bc_r=round(pf["r"], 4),
            noise_sigma=round(A.noise_sigma(agrey), 3), noise_sigma_src=round(noise_src, 3),
            jpeg_blockiness=round(A.jpeg_blockiness(agrey), 4),
            polygon_problems=";".join(probs),
            crop=os.path.relpath(A.crop_path(i, n), A.HERE), image=A.aug_image_path(i, n),
        )
        rows.append(row)
    return rows


def transform_presence(df):
    """Which of the 2024 Compose steps fired, where recoverable (NaN = not recoverable)."""
    diag = np.hypot(df.img_w, df.img_h)
    df["t_rot90"] = df.k_deg != 0
    df["t_persp"] = 1000 * df.persp_rms_px / diag > PERSP_PERMILLE
    ssr = ((df.affine_scale - 1).abs() > 5e-4) | (df.shift_px > 0.5)
    df["t_ssr"] = ssr.where(~df.t_persp)                    # confounded with Perspective's resize
    bc = ((df.bc_gain - 1).abs() >= BC_GAIN_TOL) | (df.bc_offset.abs() >= BC_OFFSET_TOL)
    df["t_bc"] = bc.where(df.bc_gain.notna())
    df["noise_ratio"] = (df.noise_sigma / df.noise_sigma_src).round(2)
    df["t_noise"] = df.noise_ratio > NOISE_RATIO
    thr = np.where(df.img_w >= 2000, BLOCK_BIG, BLOCK_SMALL)
    df["t_jpeg"] = pd.Series(df.jpeg_blockiness > thr, index=df.index).where(~df.t_noise)   # masked by noise
    for c in ("t_ssr", "t_bc", "t_jpeg"):
        df[c] = df[c].astype("boolean")
    return df


def missing_report(present):
    """Reconcile 900 generated -> 866 (notebook cells 69/72) -> 853 on disk."""
    nb = json.load(open(NOTEBOOK))
    txt = lambda c: "".join("".join(o.get("text", "")) if "text" in o else
                            "".join(o.get("data", {}).get("text/plain", "")) for o in c.get("outputs", []))
    nonconvex = set(re.findall(r"(\d+_aug\d)\.xml: Polygon is not convex", txt(nb["cells"][62])))
    low_area = set(re.findall(r"augmented_images/(\d+_aug\d)\.jpeg", txt(nb["cells"][86])))
    allids = {A.aug_id(i, n) for i in range(150) for n in range(A.N_AUG_PER_SOURCE)}
    pres = {A.aug_id(i, n) for i, n in present}
    rows = []
    for a in sorted(allids - pres, key=A.parse_aug_id):
        why = ("non-convex polygon; deleted in notebook cells 62-67" if a in nonconvex else
               "window < 21 % of the frame; deleted in notebook cells 82-88" if a in low_area else "unknown")
        rows.append(dict(aug_id=a, source_id=A.parse_aug_id(a)[0], reason=why))
    still = sorted(low_area & pres, key=A.parse_aug_id)
    return pd.DataFrame(rows), still, len(nonconvex), len(low_area)


def suggest(r):
    f = set(r["flags"].split(";")) - {""}
    if "stale_keypoints" in f:
        return ("reject", f"polygon corrupt: stored point(s) {r.stale_points} are the source's untransformed "
                          "coordinates (albumentations dropped an off-frame key point)")
    if f & {"non_convex", "out_of_bounds", "xml_size_mismatch", "not_4_points"}:
        return "reject", "invalid polygon"
    if "k_disagree" in f or "geo_inconsistent" in f:
        return "review", "geometric and photometric k disagree; inspect the contact sheet"
    if r.agree and r.geo_resid_deg < 5 and r.t_noise:
        return ("accept-if-noise-ok", f"k certain (geometric residual {r.geo_resid_deg:.1f} deg, NCC peak at the "
                                      f"same k); low NCC explained by GaussNoise (sigma x{r.noise_ratio:.0f}). "
                                      "Label valid, but the crop is very degraded")
    if r.agree and r.geo_resid_deg < 5:
        return "accept", "k certain geometrically; NCC low or ambiguous only"
    return "review", "inspect the contact sheet"


# ---------------------------------------------------------------------------- contact sheets
def _thumb(img, t=150):
    return cv2.resize(img, (t, t), interpolation=cv2.INTER_AREA)


def _label(img, s, color=(255, 255, 255)):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 16), (0, 0, 0), -1)
    cv2.putText(out, s, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
    return out


def contact_sheet(df, path, title, t=150):
    """One row per augmentation: source crop | aug crop | source rotated by k_geo | by k_ncc."""
    if not len(df):
        img = np.full((60, 600, 3), 255, np.uint8)
        cv2.putText(img, f"{title}: nothing to show", (8, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        cv2.imwrite(path, img)
        return
    rows = []
    for _, r in df.iterrows():
        src = cv2.imread(os.path.join(A.TOOL_CROPS, f"{r.source_id}.jpg"))
        aug = cv2.imread(os.path.join(A.HERE, r.crop))
        tiles = [_label(_thumb(src, t), f"src {r.source_id}"),
                 _label(_thumb(aug, t), f"{r.aug_id}", (0, 255, 255)),
                 _label(_thumb(np.rot90(src, r.k_geo_deg // 90), t), f"src rot k_geo={r.k_geo_deg}"),
                 _label(_thumb(np.rot90(src, r.k_photo_deg // 90), t), f"src rot k_ncc={r.k_photo_deg}")]
        info = np.full((t, 360, 3), 255, np.uint8)
        lines = [f"{r.aug_id}  agree={r.agree}  usable={r.usable}",
                 f"ncc j0/90/180/270: {r.ncc_j0:.2f} {r.ncc_j90:.2f} {r.ncc_j180:.2f} {r.ncc_j270:.2f}",
                 f"margin {r.ncc_margin:.3f}  geo resid {r.geo_resid_deg:.1f} deg",
                 f"S {r.rect_size} (src {r.source_rect_size})  persp {r.persp_rms_px:.1f}px  noise x{r.noise_ratio:.1f}",
                 f"flags: {r['flags'] or '-'}"[:58]]
        for q, s in enumerate(lines):
            cv2.putText(info, s, (6, 22 + 26 * q), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
        rows.append(np.hstack(tiles + [info]))
    head = np.full((34, rows[0].shape[1], 3), 255, np.uint8)
    cv2.putText(head, title, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    sheet = np.vstack([head] + [np.vstack([r, np.full((4, r.shape[1], 3), 200, np.uint8)]) for r in rows])
    cv2.imwrite(path, sheet)


MANIFEST_COLS = [
    # required by the task, first
    "aug_id", "source_id", "k_deg", "swap_hv", "polygon", "rect_size", "ncc_margin", "agree", "usable",
    "flags", "needs_review",
    # k recovery detail
    "aug_index", "k_geo_deg", "k_photo_deg", "geo_resid_deg", "geo_slot_src", "geo_slot_aug",
    "ncc_best", "ncc_second", "ncc_clahe_best", "ncc_grad_best", "ncc_j0", "ncc_j90", "ncc_j180", "ncc_j270",
    "stored_order_best_deg", "stored_order_ncc0",
    # geometry
    "polygon_canonical", "stale_points", "img_w", "img_h", "source_rect_size", "rect_ratio", "area_pct",
    "persp_rms_px", "affine_scale", "aniso", "shift_px",
    # photometry / quality
    "bc_gain", "bc_offset", "bc_r", "noise_sigma", "noise_sigma_src", "noise_ratio", "jpeg_blockiness",
    # recovered transform presence
    "t_rot90", "t_persp", "t_ssr", "t_bc", "t_noise", "t_jpeg", "nb_low_area_listed",
    "crop", "image",
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--no-crops", action="store_true", help="do not (re)write crops/")
    a = ap.parse_args()
    t0 = time.time()
    present = A.list_aug_ids()
    by_src = {}
    for i, n in present:
        by_src.setdefault(i, []).append(n)
    jobs = [(i, ns, not a.no_crops) for i, ns in sorted(by_src.items())]
    with Pool(a.workers) as p:
        res = p.map(process_source, jobs, chunksize=1)
    df = pd.DataFrame([r for rr in res for r in rr])
    df = df.sort_values(["source_id", "aug_index"]).reset_index(drop=True)
    df = transform_presence(df)

    flags = []
    for _, r in df.iterrows():
        f = [x for x in r.polygon_problems.split(";") if x]
        if not r.agree:
            f.append("k_disagree")
        if r.ncc_best < NCC_MIN:
            f.append("ncc_low")
        if r.ncc_margin < MARGIN_MIN:
            f.append("margin_low")
        if r.geo_resid_deg > GEO_RESID_MAX:
            f.append("geo_inconsistent")
        flags.append(";".join(f))
    df["flags"] = flags
    df["usable"] = df["flags"] == ""
    df["needs_review"] = ~df.usable

    miss, still_low, n_nc, n_low = missing_report(present)
    df["nb_low_area_listed"] = df.aug_id.isin(still_low)
    df[MANIFEST_COLS].to_csv(A.MANIFEST, index=False)
    miss.to_csv(os.path.join(A.HERE, "missing_augmentations.csv"), index=False)

    os.makedirs(REVIEW_DIR, exist_ok=True)
    fl = df[~df.usable].copy()
    sug = [suggest(r) for _, r in fl.iterrows()]
    fl["suggested"] = [s[0] for s in sug]
    fl["reason"] = [s[1] for s in sug]
    dec_path = os.path.join(REVIEW_DIR, "review_decisions.csv")
    old = pd.read_csv(dec_path, dtype=str, keep_default_na=False) if os.path.exists(dec_path) else None
    out = fl[["aug_id", "source_id", "k_geo_deg", "k_photo_deg", "ncc_best", "ncc_margin", "flags",
              "suggested", "reason"]].copy()
    out["decision"] = ""          # accept | reject   (human; empty = follow manifest `usable`)
    out["k_deg_override"] = ""    # optional, CCW degrees, only with decision=accept
    out["reviewer_note"] = ""
    if old is not None and len(old):  # keep human entries across rebuilds
        keep = old.set_index("aug_id")
        for c in ("decision", "k_deg_override", "reviewer_note"):
            if c in keep:
                out[c] = out.aug_id.map(keep[c]).fillna("")
    out.to_csv(dec_path, index=False)
    contact_sheet(fl, os.path.join(REVIEW_DIR, "contact_flagged.png"),
                  f"NOT auto-accepted: {len(fl)} of {len(df)}   (source | aug | source rotated by k_geo | by k_ncc)")
    ok = df[df.usable].sample(n=min(24, int(df.usable.sum())), random_state=0).sort_values(["source_id", "aug_index"])
    contact_sheet(ok, os.path.join(REVIEW_DIR, "contact_ok_sample.png"),
                  f"random 24 of {int(df.usable.sum())} accepted   (source | aug | source rotated by k_geo | by k_ncc)")

    print(f"{len(df)} augmentations from {df.source_id.nunique()} sources; {time.time() - t0:.0f}s")
    print("missing on disk:", len(miss), miss.reason.value_counts().to_dict(),
          f"| notebook lists {n_nc} non-convex and {n_low} low-area ids ({len(still_low)} low-area still on disk)")
    print("k (CCW) distribution:", df.k_deg.value_counts().sort_index().to_dict())
    print(f"agree (geo == ncc): {int(df.agree.sum())}/{len(df)}")
    print("stored-order rectification best rotation:", df.stored_order_best_deg.value_counts().to_dict())
    print("ncc_best quantiles:", df.ncc_best.quantile([0, .01, .05, .5]).round(3).to_dict())
    print("ncc_margin quantiles:", df.ncc_margin.quantile([0, .01, .05, .5]).round(3).to_dict())
    print("usable:", int(df.usable.sum()), "flag counts:",
          pd.Series([x for f in df["flags"] for x in f.split(";") if x]).value_counts().to_dict())
    print("transform presence (True / False / unknown):")
    for c in ("t_rot90", "t_persp", "t_ssr", "t_bc", "t_noise", "t_jpeg"):
        s = df[c].astype("boolean")
        print(f"  {c:8s} {int(s.sum())} / {int((s == False).sum())} / {int(s.isna().sum())}")  # noqa: E712
    print(fl[["aug_id", "flags", "suggested"]].to_string(index=False))


if __name__ == "__main__":
    main()
