#!/usr/bin/env python
"""verify_augment.py -- empirical label-invariance check of every augment.py transform.

No human labels are used.  For each image, each transform x strength is applied and the frozen
classical counter ("ours", methods/results/counting_frozen_final.py) and the existing synthetic-only
CNN (runs/synth_r18_v2mix_aniso/model.pt, standard 3-scale x 3-flip TTA) are run on the result.
Drift = observed prediction change / change predicted by the transform's LabelTransform - 1, per
image and axis (h<->v swapped where the transform swaps them).

Sets
  real150   the 150 rectified annotation crops (annotation_tool/data/crops/{i}.jpg)      -> DECISIONS
  synth100  first 100 images of cnn/data/synth_val (exact labels -> also error vs the exactly
            transformed label)                                                         -> DECISIONS
  legacy13  the 13 canonical legacy crops (vlm/crops/legacy/*.png).  Label-free drift only,
            REPORTED but never used for any decision (legacy-13 is the held-out test set).
All images are capped at 1024 px (cnn_common.cap_side, as in training) before transforming.

Conventions
  ours measures crossing counts per crop side (h_count = k_x of the lattice), so its expected change
  is lt.crossing() with the yarn tilt from ours' clean run; the CNN was trained on intrinsic
  densities, so its primary expectation is lt.density().  Both are stored for both models.
  ours is only scored where the expected per-crop count is inside its search band
  [1.1 R_MIN, 0.40 N_work] (R_MIN = 18 threads per crop side, N_work = min(side, 1024)).

Noise floor and PASS rule (fixed before any transform was run)
  floor = pooled drift of the two null re-encodings (JPEG q95 re-encode; 0.5-px sub-pixel shift),
  same set / model / rows.  A transform x strength is 'within clean-run noise' iff ALL of
     median |drift|       <= floor median + 0.5 pp
     p90 |drift|          <= floor p90    + 2.0 pp
     P(|drift| > 10 %)    <= floor rate   + 2.0 pp
     |median signed drift| <= 1.0 %
  label_preserving (the whitelist flag) = PASS for ours on real150 AND on synth100.  For size-changing
  transforms (zoom crop, tiles) a size-matched floor (transform + JPEG q95 vs transform) is also
  reported, because ours is intrinsically noisier on smaller windows.
  Secondary diagnostic (added after the 6-image smoke test, before the full run; never changes the
  primary flag): the same rule restricted to images where ours' clean-run confidence >= 0.5
  (PASS_conf), because ours can flip between lattice hypotheses on ambiguous images under ANY
  perturbation, which says more about ours than about the transform.
  Tier B (defined after the smoke test and a partial look at 41 real-150 images, before the full
  results; legacy-13 never consulted): a transform whose label map is exact by construction
  (unit-tested: flips, rot90, inner rotation, zoom, tiles, perspective; or pixel-wise photometric /
  sensor / illumination) that fails the strict rule but is UNBIASED on both instruments
  (|median signed drift| <= 1 % for ours and <= 2 % for the CNN, on real-150 and synth-100).
  Rationale: on these transforms ours' tails come from discrete lattice-hypothesis switches under
  resampling / field-of-view changes (rot90 flips 6 % of values on ambiguous images), i.e. from the
  instrument, while a wrong label map shows up as a systematic bias (the label-agnostic resized crop:
  -33 %).  augment_whitelist.json 'policy' = tier A only (default); 'policy_extended' = A + B.
  Also reported on synth-100: accuracy vs the EXACTLY transformed label compared with clean accuracy
  (paired cluster bootstrap), and a paired one-sided Wilcoxon test of per-image |drift| vs the floor
  (Holm-adjusted within set x model).

  python verify_augment.py ours [--workers 8]
  python verify_augment.py cnn
  python verify_augment.py summarise
"""
from __future__ import annotations

import os as _os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_k, "1")

import argparse
import importlib.util
import json
import math
import os
import sys
import time
import zlib

import cv2
import numpy as np
import pandas as pd

cv2.setNumThreads(1)
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import augment as A  # noqa: E402

OURS_FILE = os.path.join(ROOT, "methods", "results", "counting_frozen_final.py")
CONF_FILE = os.path.join(ROOT, "methods", "conf_model.json")
OUT = os.environ.get("AUGV_OUT", os.path.join(HERE, "aug_verify"))
LIMIT = int(os.environ.get("AUGV_LIMIT", "0"))   # smoke tests: first N images per set
DEST = HERE if "AUGV_OUT" not in os.environ else OUT   # final tables / figures (smoke tests stay in OUT)
CKPT = os.path.join(HERE, "runs", "synth_r18_v2mix_aniso", "model.pt")
TOOL_CROPS = os.path.join(ROOT, "annotation_tool", "data", "crops")
SYNTH_VAL = os.path.join(HERE, "data", "synth_val")
LEGACY_CROPS = os.path.join(ROOT, "vlm", "crops", "legacy")
LEGACY_NAMES_CSV = os.path.join(HERE, "runs", "synth_r18_v2mix_aniso", "eval_vlmcrops", "legacy_predictions_vlm.csv")
R_MIN, NYQ = 18.0, 0.42
RAW_COLS = ["set", "image", "cond", "variant", "model", "side", "pred_h", "pred_v", "pred_weave", "conf", "angle_x",
            "angle_y", "n_work", "runtime_s", "swap", "fc_x", "fc_y", "f0_x", "f0_y", "fd_x", "fd_y", "ext_x", "ext_y",
            "ppi_x", "ppi_y", "out_w", "out_h", "status", "lab_cross_h", "lab_cross_v", "lab_dens_h", "lab_dens_v"]
BAND_LO, BAND_HI = 1.1 * R_MIN, 0.40


# =============================================================================================
# data
# =============================================================================================
def cap_side(img, max_side=1024):
    """== cnn_common.cap_side (copied so the CPU workers do not import torch)."""
    if img.shape[0] > max_side or img.shape[1] > max_side:
        return cv2.resize(img, (max_side, max_side), interpolation=cv2.INTER_AREA)
    if img.shape[0] != img.shape[1]:
        s = max(img.shape[:2])
        return cv2.resize(img, (s, s), interpolation=cv2.INTER_CUBIC)
    return img


def load_sets(names=("real150", "synth100", "legacy13")):
    """{set: [dict(image, path, [exact labels])]}"""
    out = {}
    if "real150" in names:
        out["real150"] = [dict(image=str(i), path=os.path.join(TOOL_CROPS, f"{i}.jpg")) for i in range(150)]
    if "synth100" in names:
        df = pd.read_csv(os.path.join(SYNTH_VAL, "labels.csv")).iloc[:100]
        out["synth100"] = [dict(image=os.path.splitext(r.file)[0], path=os.path.join(SYNTH_VAL, r.file),
                                epi=float(r.h_count), ppi=float(r.v_count), rot=float(r.rotation_deg),
                                skew=float(r.skew_deg), weave_coarse=r.weave_coarse) for r in df.itertuples()]
    if "legacy13" in names:
        nm = pd.read_csv(LEGACY_NAMES_CSV, usecols=["name"])["name"].tolist()   # names only, no GT
        out["legacy13"] = [dict(image=n, path=os.path.join(LEGACY_CROPS, f"{n}.png")) for n in nm]
    if LIMIT:
        out = {k: v[:LIMIT] for k, v in out.items()}
    return out


def load_img(path):
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(path)
    return cap_side(im, 1024)


# =============================================================================================
# conditions
# =============================================================================================
def _c(cond, family, kind, fn, level, **params):
    return dict(cond=cond, family=family, kind=kind, fn=fn, strength=level, params=params)


def conditions():
    C = [_c("clean", "clean", "clean", "identity", 0)]
    C += [_c("null_jpeg95", "null", "null", "jpeg", 95, q=95),
          _c("null_subpix", "null", "null", "subpix", 0.5)]
    C += [_c("hflip", "a_flip", "candidate", "hflip", 1), _c("vflip", "a_flip", "candidate", "vflip", 1)]
    C += [_c("rot90", "b_rot90", "candidate", "rot90", 90, k=1), _c("rot270", "b_rot90", "candidate", "rot90", 270, k=3)]
    for d in (-15, -10, -5, -2, 2, 5, 10, 15):
        C.append(_c(f"rot_inner_{d:+d}", "c_rotate", "candidate", "rotate", d, deg=float(d), mode="inner"))
    for d in (5, 15):
        C.append(_c(f"rot_reflect_{d:+d}", "c_rotate", "candidate", "rotate", d, deg=float(d), mode="reflect"))
    C += [_c("bright_0.6", "d_photometric", "candidate", "brightness", 0.6, factor=0.6),
          _c("bright_1.4", "d_photometric", "candidate", "brightness", 1.4, factor=1.4),
          _c("contrast_0.6", "d_photometric", "candidate", "contrast", 0.6, factor=0.6),
          _c("contrast_1.5", "d_photometric", "candidate", "contrast", 1.5, factor=1.5),
          _c("gamma_0.6", "d_photometric", "candidate", "gamma", 0.6, g=0.6),
          _c("gamma_1.7", "d_photometric", "candidate", "gamma", 1.7, g=1.7),
          _c("hue_+60", "d_photometric", "candidate", "hue_shift", 60, deg=60.0),
          _c("sat_0.25", "d_photometric", "candidate", "saturation", 0.25, factor=0.25),
          _c("sat_1.75", "d_photometric", "candidate", "saturation", 1.75, factor=1.75),
          _c("greyscale", "d_photometric", "candidate", "greyscale", 1),
          _c("clahe_2", "d_photometric", "candidate", "clahe", 2, clip=2.0),
          _c("clahe_4", "d_photometric", "candidate", "clahe", 4, clip=4.0),
          _c("wb_warm", "d_photometric", "candidate", "white_balance", 1.25, gains=(0.75, 1.0, 1.25)),
          _c("wb_cool", "d_photometric", "candidate", "white_balance", 0.75, gains=(1.25, 1.0, 0.75))]
    C += [_c(f"noise_{s:g}", "e_sensor", "candidate", "gaussian_noise", s, sigma=s) for s in (0.01, 0.03, 0.06)]
    C += [_c(f"jpeg_{q}", "e_sensor", "candidate", "jpeg", q, q=q) for q in (70, 40, 20)]
    C += [_c(f"blur_{s:g}px", "e_sensor", "candidate", "gaussian_blur", s, sigma=s) for s in (0.5, 1.0, 1.5)]
    C += [_c(f"blur_{f:g}pitch", "e_sensor", "candidate", "blur_pitch", f, frac=f) for f in (0.1, 0.2)]
    C += [_c(f"motion_{L}px", "e_sensor", "candidate", "motion_blur", L, length=L) for L in (3, 7)]
    C += [_c(f"sharpen_{a:g}", "e_sensor", "candidate", "sharpen", a, amount=a) for a in (1.0, 2.5)]
    C += [_c("glare_mild", "f_illumination", "candidate", "glare", 1, n_blobs=2, amp=0.4),
          _c("glare_strong", "f_illumination", "candidate", "glare", 2, n_blobs=4, amp=1.0),
          _c("shadow_soft", "f_illumination", "candidate", "shadow", 0.5, depth=0.5),
          _c("shadow_deep", "f_illumination", "candidate", "shadow", 0.15, depth=0.15),
          _c("vignette_0.3", "f_illumination", "candidate", "vignette", 0.3, strength=0.3),
          _c("vignette_0.6", "f_illumination", "candidate", "vignette", 0.6, strength=0.6)]
    C += [_c(f"zoom_{s:g}", "g_zoom", "candidate", "zoom", s, s=s) for s in (0.5, 0.7, 1.25, 1.6, 2.0)]
    C += [_c("zoomxy_1.25x1", "g_zoom", "candidate", "zoom_xy", 1.25, sx=1.25, sy=1.0),
          _c("zoomxy_1x0.8", "g_zoom", "candidate", "zoom_xy", 0.8, sx=1.0, sy=0.8)]
    C += [_c(f"tile_{t:g}in", "h_tile", "candidate", "tiles", t, size_in=t) for t in (0.5, 0.25)]
    C += [_c(f"persp_{e:g}", "i_perspective", "candidate", "perspective", e, eps=e) for e in (0.01, 0.02, 0.03)]
    # size-matched noise floors (transform followed by a JPEG q95 re-encode)
    C += [_c(f"{b}+jpeg95", "z_sizefloor", "sizefloor", "then_jpeg95", 0, base=b)
          for b in ("zoom_0.5", "zoom_0.7", "zoom_1.6", "zoom_2", "tile_0.5in", "tile_0.25in", "rot_inner_+15")]
    # negative controls (EXCLUDED families)
    C += [_c("elastic_0.01", "x_excluded", "excluded", "elastic", 0.01, alpha=0.01, sigma=0.05),
          _c("elastic_0.03", "x_excluded", "excluded", "elastic", 0.03, alpha=0.03, sigma=0.05),
          _c("mixup_0.5", "x_excluded", "excluded", "mixup", 0.5, lam=0.5),
          _c("cutmix_0.5", "x_excluded", "excluded", "cutmix", 0.5, frac=0.5),
          _c("blur_0.4pitch", "x_excluded", "excluded", "strong_blur_pitch", 0.4, frac=0.4),
          _c("naiveRRC_1.4", "x_excluded", "excluded", "label_agnostic_resized_crop", 1.4, s=1.4),
          _c("rot_inner_+30", "x_excluded", "excluded", "rotate", 30, deg=30.0, mode="inner")]
    return C


COND = {c["cond"]: c for c in conditions()}


def subpix(img, extent=(1.0, 1.0), rng=None, d=0.5):
    H, W = img.shape[:2]
    M = np.array([[1, 0, d], [0, 1, d]], float)
    out = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT_101)
    return out, A._affine_lt("subpix", img, (W, H), M, extent)


def variants(c, img, seed_key, pitch_px, partner):
    """[(variant_id, image, LabelTransform)] for condition c."""
    rng = np.random.default_rng(zlib.crc32(seed_key.encode()))
    fn, p = c["fn"], c["params"]
    if fn == "identity":
        return [(0, img, A.identity_lt(img))]
    if fn == "subpix":
        return [(0,) + subpix(img, d=c["strength"])]
    if fn == "tiles":
        return [(k, t, lt) for k, (t, lt) in enumerate(A.tiles(img, size_in=p["size_in"]))]
    if fn == "blur_pitch":
        return [(0,) + A.gaussian_blur(img, sigma=p["frac"] * pitch_px)]
    if fn == "strong_blur_pitch":
        return [(0,) + A.strong_blur(img, sigma=p["frac"] * pitch_px)]
    if fn in ("mixup", "cutmix"):
        return [(0,) + A.TRANSFORMS[fn](img, rng=rng, partner=partner, **p)]
    if fn == "then_jpeg95":
        base = COND[p["base"]]
        out = []
        for k, t, lt in variants(base, img, seed_key.replace(c["cond"], base["cond"]), pitch_px, partner):
            t2, lt2 = A.jpeg(t, extent=lt.extent_out, q=95)
            out.append((k, t2, A.compose(lt, lt2)))
        return out
    if fn in ("zoom", "zoom_xy"):
        p = dict(p, center=(0.5, 0.5))
    return [(0,) + A.TRANSFORMS[fn](img, rng=rng, **p)]


def lt_columns(lt, angles, synth=None):
    """Label-transform factors stored with every prediction row."""
    if lt.params.get("true_lt") is not None:      # naive RRC: what it CLAIMS is lt, the truth is true_lt
        true_lt = lt.params["true_lt"]
    else:
        true_lt = lt
    hc, vc, _ = lt.crossing((1.0, 1.0), angles)
    h0, v0, _ = lt.crossing((1.0, 1.0), (0.0, 0.0))
    hd, vd, _ = lt.density((1.0, 1.0), angles, kind="density")
    row = dict(swap=bool(lt.swaps_axes), fc_x=hc, fc_y=vc, f0_x=h0, f0_y=v0, fd_x=hd, fd_y=vd,
               ext_x=lt.extent_out[0], ext_y=lt.extent_out[1], ppi_x=lt.ppi_out[0], ppi_y=lt.ppi_out[1],
               out_w=lt.out_size[0], out_h=lt.out_size[1], status=lt.status)
    if synth is not None:   # exact labels of the transformed image (intrinsic epi/ppi + exact tilt)
        ang = (synth["rot"], -(synth["rot"] + synth["skew"]))
        lab = (synth["epi"], synth["ppi"])
        ch, cv, _ = true_lt.crossing(lab, ang, kind="density")
        dh, dv, _ = true_lt.density(lab, ang, kind="density")
        row.update(lab_cross_h=ch, lab_cross_v=cv, lab_dens_h=dh, lab_dens_v=dv)
    return row


# =============================================================================================
# ours (CPU pool)
# =============================================================================================
_O = None


def _init_ours():
    global _O
    spec = importlib.util.spec_from_file_location("ours_frozen", OURS_FILE)
    _O = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_O)
    # the frozen copy looks for conf_model.json next to itself; use the fitted model of the release
    _O.CONF_MODEL = json.load(open(CONF_FILE))


def _ours(img):
    t0 = time.time()
    e = _O.estimate(img, "ours")
    lat = e["debug"].get("lattice") or {}
    return dict(pred_h=e["h_count"], pred_v=e["v_count"], pred_weave=e["weave"], conf=e["confidence"],
                angle_x=lat.get("angle_x", np.nan), angle_y=lat.get("angle_y", np.nan),
                n_work=e["debug"].get("N_work", np.nan), runtime_s=time.time() - t0)


def ours_job(a):
    sname, item, partner_path, clean, conds = a
    img = load_img(item["path"])
    partner = load_img(partner_path) if partner_path else None
    S = img.shape[0]
    rows = []
    ang = (0.0, 0.0)
    pitch = S / 60.0
    if clean is not None:
        ax, ay = clean["angle_x"], clean["angle_y"]
        ang = (0.0 if not np.isfinite(ax) else ax, 0.0 if not np.isfinite(ay) else ay)
        m = np.nanmax([clean["pred_h"], clean["pred_v"]])
        pitch = S / m if np.isfinite(m) and m > 0 else pitch
    synth = item if sname == "synth100" else None
    for cn in conds:
        c = COND[cn]
        for k, t, lt in variants(c, img, f"{sname}|{item['image']}|{cn}", pitch, partner):
            r = dict(set=sname, image=item["image"], cond=cn, variant=k, model="ours", side=S)
            r.update(_ours(t))
            r.update(lt_columns(lt, ang, synth))
            rows.append(r)
    return rows


def run_ours(sets, workers, conds=None):
    from multiprocessing import Pool
    os.makedirs(OUT, exist_ok=True)
    conds = conds or [c["cond"] for c in conditions()]
    clean_path = os.path.join(OUT, "clean_ours.csv")
    S = load_sets(sets)
    t0 = time.time()
    # stage 1: clean run (gives yarn tilt for the crossing expectation and the pitch for pitch-scaled blur)
    if not os.path.exists(clean_path):
        jobs = [(s, it, None, None, ["clean"]) for s in S for it in S[s]]
        with Pool(workers, initializer=_init_ours) as pool:
            rows = [r for rr in pool.imap_unordered(ours_job, jobs, chunksize=2) for r in rr]
        pd.DataFrame(rows).reindex(columns=RAW_COLS).to_csv(clean_path, index=False)
        print(f"clean ours: {len(rows)} rows, {time.time() - t0:.0f}s", flush=True)
    clean = pd.read_csv(clean_path, dtype={"image": str}).set_index(["set", "image"])
    raw = os.path.join(OUT, "raw_ours.csv")
    done = set()
    if os.path.exists(raw):
        d = pd.read_csv(raw, usecols=["set", "image"], dtype={"image": str})
        done = set(zip(d["set"], d["image"]))
    jobs = []
    for s in S:
        n = len(S[s])
        for j, it in enumerate(S[s]):
            if (s, it["image"]) in done:
                continue
            jobs.append((s, it, S[s][(j + 1) % n]["path"], clean.loc[(s, it["image"])].to_dict(), conds))
    print(f"ours: {len(jobs)} images x {len(conds)} conditions, {workers} workers", flush=True)
    header = not os.path.exists(raw)
    k = 0
    with Pool(workers, initializer=_init_ours) as pool:
        for rows in pool.imap_unordered(ours_job, jobs, chunksize=1):
            pd.DataFrame(rows).reindex(columns=RAW_COLS).to_csv(raw, mode="a", header=header, index=False)
            header = False
            k += 1
            if k % 10 == 0 or k == len(jobs):
                print(f"  ours {k}/{len(jobs)} images  {time.time() - t0:.0f}s", flush=True)


# =============================================================================================
# CNN (MPS)
# =============================================================================================
def _tile_pos(n, b, T, stride):
    hi = n - b - T
    if hi < b:
        return [max(0, (n - T) // 2)]
    pos = list(range(b, hi + 1, stride))
    if pos[-1] != hi:
        pos.append(hi)
    return pos


def predict_phys(model, img, ppi, device, Ps=(672, 768, 864), flips=True, stride=112, batch=192):
    """Per-inch (h, v) for an image of known px/inch, == cnn_common.predict_crop for a full 1-inch crop
    (same resampling, 3.5 % border, tile grid, flips, median of log tile counts); sub-inch images
    (zoom crops, inner rotations, tiles) are resampled to the same P px/inch with no border, and if a
    224-px tile does not fit at any test P the smallest P that fits is used."""
    import torch
    import cnn_common as C
    H, W = img.shape[:2]
    ex, ey = W / ppi[0], H / ppi[1]
    full = ex > 0.999 and ey > 0.999
    bf = C.BORDER if full else 0.0
    use = [P for P in Ps if min(ex, ey) * P - 2 * int(bf * P) >= C.TILE]
    if not use:
        use = [int(math.ceil(C.TILE / min(ex, ey)))]
    mean = torch.tensor(C.IMAGENET_MEAN, device=device).view(3, 1, 1)
    std = torch.tensor(C.IMAGENET_STD, device=device).view(3, 1, 1)
    logs, wps = [], []
    with torch.no_grad():
        for P in use:
            if full:
                Wp = Hp = P
                interp = cv2.INTER_AREA if img.shape[0] > P else cv2.INTER_CUBIC
            else:
                Wp, Hp = max(C.TILE, int(round(ex * P))), max(C.TILE, int(round(ey * P)))
                interp = cv2.INTER_AREA if (W > Wp and H > Hp) else cv2.INTER_CUBIC
            im = cv2.resize(img, (Wp, Hp), interpolation=interp)
            x = torch.from_numpy(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)).to(device).permute(2, 0, 1).float() / 255.0
            x = (x - mean) / std
            b = int(bf * P)
            ys, xs = _tile_pos(Hp, b, C.TILE, stride), _tile_pos(Wp, b, C.TILE, stride)
            tiles = torch.stack([x[:, y:y + C.TILE, xx:xx + C.TILE] for y in ys for xx in xs])
            views = [tiles, tiles.flip(-1), tiles.flip(-2)] if flips else [tiles]
            for v in views:
                for k in range(0, len(v), batch):
                    pc, pw = model(v[k:k + batch])
                    logs.append(pc.float().cpu().numpy() + math.log(P / C.TILE))
                    wps.append(torch.softmax(pw.float(), -1).cpu().numpy())
    L = np.concatenate(logs)
    Wq = np.concatenate(wps).mean(0)
    hv = np.exp(np.median(L, 0))
    return float(hv[0]), float(hv[1]), C.WEAVE_CLASSES[int(np.argmax(Wq))], Wq


def run_cnn(sets, conds=None, ckpt=CKPT, check_clean=True):
    import torch  # noqa: F401
    import cnn_common as C
    os.makedirs(OUT, exist_ok=True)
    dev = C.get_device("auto")
    model = C.load_ckpt(ckpt, dev)
    conds = conds or [c["cond"] for c in conditions()]
    clean = pd.read_csv(os.path.join(OUT, "clean_ours.csv"), dtype={"image": str}).set_index(["set", "image"])
    S = load_sets(sets)
    raw = os.path.join(OUT, "raw_cnn_" + "+".join(sets) + ".csv")
    done = set()
    if os.path.exists(raw):
        d = pd.read_csv(raw, usecols=["set", "image"], dtype={"image": str})
        done = set(zip(d["set"], d["image"]))
    header = not os.path.exists(raw)
    t0 = time.time()
    n_tot = sum(len(v) for v in S.values())
    k = 0
    for s in S:
        n = len(S[s])
        for j, it in enumerate(S[s]):
            k += 1
            if (s, it["image"]) in done:
                continue
            img = load_img(it["path"])
            partner = load_img(S[s][(j + 1) % n]["path"])
            cl = clean.loc[(s, it["image"])]
            ax, ay = cl["angle_x"], cl["angle_y"]
            ang = (0.0 if not np.isfinite(ax) else ax, 0.0 if not np.isfinite(ay) else ay)
            m = np.nanmax([cl["pred_h"], cl["pred_v"]])
            pitch = img.shape[0] / m if np.isfinite(m) and m > 0 else img.shape[0] / 60.0
            rows = []
            for cn in conds:
                for vk, t, lt in variants(COND[cn], img, f"{s}|{it['image']}|{cn}", pitch, partner):
                    t1 = time.time()
                    h, v, w, wp = predict_phys(model, t, lt.ppi_out, dev)
                    r = dict(set=s, image=it["image"], cond=cn, variant=vk, model="cnn", side=img.shape[0],
                             pred_h=h, pred_v=v, pred_weave=w, conf=float(wp.max()), runtime_s=time.time() - t1)
                    r.update(lt_columns(lt, ang, it if s == "synth100" else None))
                    rows.append(r)
            pd.DataFrame(rows).reindex(columns=RAW_COLS).to_csv(raw, mode="a", header=header, index=False)
            header = False
            if k % 10 == 0:
                print(f"  cnn {k}/{n_tot} images  {time.time() - t0:.0f}s", flush=True)
    if check_clean:   # predict_phys must reproduce evaluate.py's real150 predictions on clean crops
        ref = os.path.join(HERE, "runs", "synth_r18_v2mix_aniso", "eval_final", "real150_predictions.csv")
        if os.path.exists(ref):
            d = pd.read_csv(raw, dtype={"image": str})
            d = d[(d.set == "real150") & (d.cond == "clean")].copy()
            d["image"] = d["image"].astype(int)
            m = d.merge(pd.read_csv(ref), on="image", suffixes=("", "_ref"))
            if not len(m):        # this process had no real-150 images
                return
            e = np.abs(np.log(np.r_[m.pred_h / m.pred_h_ref, m.pred_v / m.pred_v_ref]))
            print(f"clean CNN vs eval_final/real150_predictions.csv: n={len(m)}, max |log ratio| = {e.max():.2e}, "
                  f"median {np.median(e):.2e}", flush=True)


# =============================================================================================
# summary
# =============================================================================================
OURS_W = {"plain": "plain", "twill_2_1": "twill", "twill_3_1": "twill", "twill_2_2": "twill", "satin": "satin",
          "other": "other"}


def long_drift(df, clean):
    """One row per (image, variant, output axis) with expected and observed per-inch values."""
    d = df.merge(clean, on=["set", "image", "model"], suffixes=("", "_0"))
    out = []
    for ax in ("x", "y"):
        pred = d["pred_h"] if ax == "x" else d["pred_v"]
        same = d["pred_h_0"] if ax == "x" else d["pred_v_0"]
        other = d["pred_v_0"] if ax == "x" else d["pred_h_0"]
        base = np.where(d["swap"], other, same)            # clean value of the family landing on this axis
        ext = d[f"ext_{ax}"]
        per_inch = np.where(d["model"] == "ours", pred / ext, pred)
        n_work = np.minimum(np.maximum(d["out_w"], d["out_h"]), 1024)
        e = pd.DataFrame({"set": d["set"], "image": d["image"], "cond": d["cond"], "variant": d["variant"],
                          "model": d["model"], "axis": ax, "obs": per_inch, "base": base,
                          "fc": d[f"fc_{ax}"], "fd": d[f"fd_{ax}"], "f0": d[f"f0_{ax}"], "ext": ext,
                          "conf0": d["conf_0"], "weave": d["pred_weave"], "weave0": d["pred_weave_0"]})
        # ours: expected per-crop count inside the search band (on clean and transformed image)
        exp_crop = base * d[f"fc_{ax}"] * ext
        base_crop = base
        side0 = d["side"].clip(upper=1024)
        e["in_band"] = ((exp_crop >= BAND_LO) & (exp_crop <= BAND_HI * n_work) & (base_crop >= BAND_LO)
                        & (base_crop <= BAND_HI * side0)) | (d["model"] != "ours")
        if "lab_cross_h" in d:
            e["lab_cross"] = d["lab_cross_h"] if ax == "x" else d["lab_cross_v"]
            e["lab_dens"] = d["lab_dens_h"] if ax == "x" else d["lab_dens_v"]
            # exact clean label of the family that lands on this output axis
            hh, vv = ("h", "v") if ax == "x" else ("v", "h")
            e["lab_cross0"] = np.where(d["swap"], d[f"lab_cross_{vv}_0"], d[f"lab_cross_{hh}_0"])
            e["lab_dens0"] = np.where(d["swap"], d[f"lab_dens_{vv}_0"], d[f"lab_dens_{hh}_0"])
        out.append(e)
    e = pd.concat(out, ignore_index=True)
    e = e[np.isfinite(e.base) & (e.base > 0)].copy()      # clean run failed -> not evaluable
    e["failed"] = ~(np.isfinite(e.obs) & (e.obs > 0))
    e["drift_cross"] = e.obs / (e.fc * e.base) - 1
    e["drift_dens"] = e.obs / (e.fd * e.base) - 1
    for c in ("drift_cross", "drift_dens"):              # a failed run counts as a 100 % drift
        e.loc[e.failed, c] = 1.0
    e["primary"] = np.where(e.model == "ours", e.drift_cross, e.drift_dens)
    return e


def stats(x):
    a = np.abs(x)
    return dict(med=100 * np.median(a), p90=100 * np.percentile(a, 90), gross10=100 * np.mean(a > 0.10),
                gt5=100 * np.mean(a > 0.05), bias=100 * np.median(np.log1p(x)))


def boot_delta(e, f, B=1000, seed=0):
    """Paired cluster bootstrap over images of (stat(e) - stat(floor)) -> 95% CIs."""
    rng = np.random.default_rng(seed)
    imgs = np.union1d(e.image.unique(), f.image.unique())
    ge = {k: g.values for k, g in e.groupby("image")["primary"]}
    gf = {k: g.values for k, g in f.groupby("image")["primary"]}
    res = {k: [] for k in ("med", "p90", "gross10", "bias")}
    for _ in range(B):
        s = rng.choice(imgs, len(imgs), replace=True)
        xe = np.concatenate([ge[i] for i in s if i in ge] or [np.zeros(0)])
        xf = np.concatenate([gf[i] for i in s if i in gf] or [np.zeros(0)])
        if not len(xe) or not len(xf):
            continue
        a, b = stats(xe), stats(xf)
        for k in res:
            res[k].append(a[k] - b[k] if k != "bias" else a[k])     # bias: of the transform itself
    return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in res.items()}


def passes(s, f):
    return (s["med"] <= f["med"] + 0.5 and s["p90"] <= f["p90"] + 2.0 and s["gross10"] <= f["gross10"] + 2.0
            and abs(s["bias"]) <= 1.0)


def summarise():
    import glob
    frames = [pd.read_csv(f, dtype={"image": str}) for f in sorted(glob.glob(os.path.join(OUT, "raw_*.csv")))]
    raw = pd.concat(frames, ignore_index=True)
    clean = raw[raw.cond == "clean"][["set", "image", "model", "pred_h", "pred_v", "pred_weave", "conf"] +
                                     [c for c in ("lab_cross_h", "lab_cross_v", "lab_dens_h", "lab_dens_v")
                                      if c in raw]]
    clean = clean.rename(columns={c: c + "_0" for c in clean.columns if c not in ("set", "image", "model")})
    e = long_drift(raw, clean)
    e.to_csv(os.path.join(OUT, "drift_long.csv.gz"), index=False)
    C = {c["cond"]: c for c in conditions()}
    rows = []
    for (s, m), g in e.groupby(["set", "model"]):
        gb = g[g.in_band]
        floor_rows = gb[gb.cond.isin(["null_jpeg95", "null_subpix"])]
        fl = stats(floor_rows.primary.values)
        for cn, h in gb.groupby("cond"):
            if cn == "clean":
                continue
            c = C[cn]
            st = stats(h.primary.values)
            stc, std_ = stats(h.drift_cross.values), stats(h.drift_dens.values)
            # floor excluding the condition itself (null vs the other null, for calibration)
            fr = floor_rows[floor_rows.cond != cn] if c["kind"] == "null" else floor_rows
            f2 = stats(fr.primary.values)
            ci = boot_delta(h, fr, B=400 if c["kind"] != "null" else 400, seed=zlib.crc32(cn.encode()))
            # size-matched floor
            sz = None
            for b in ("zoom_0.5", "zoom_0.7", "zoom_1.6", "zoom_2", "tile_0.5in", "tile_0.25in", "rot_inner_+15"):
                if cn == b and (b + "+jpeg95") in set(gb.cond):
                    zz = gb[gb.cond == b + "+jpeg95"][["image", "variant", "axis", "obs"]]
                    mm = h.merge(zz, on=["image", "variant", "axis"], suffixes=("", "_j"))
                    sz = stats((mm.obs_j / mm.obs - 1).values)
            ok = passes(st, f2)
            # paired per-image test: mean |drift| of the transform vs mean |drift| of the floor (one-sided)
            pi_t = h.assign(a=h.primary.abs()).groupby("image").a.mean()
            pi_f = fr.assign(a=fr.primary.abs()).groupby("image").a.mean()
            jj = pi_t.index.intersection(pi_f.index)
            try:
                from scipy.stats import wilcoxon
                dd = (pi_t[jj] - pi_f[jj]).values
                wil_p = float(wilcoxon(dd, alternative="greater").pvalue) if np.any(dd != 0) else 1.0
            except Exception:
                wil_p = np.nan
            ok_conf, st_conf = np.nan, None
            if m == "ours":
                hc_, fc_ = h[h.conf0 >= 0.5], fr[fr.conf0 >= 0.5]
                if len(hc_) and len(fc_):
                    st_conf = stats(hc_.primary.values)
                    ok_conf = passes(st_conf, stats(fc_.primary.values))
            ok_sz = passes(st, {k: max(f2[k], sz[k]) if k != "bias" else 0 for k in f2}) if sz else ok
            all_rows = g[g.cond == cn]
            wv = all_rows.drop_duplicates(["image", "variant"])
            wmap = (lambda x: OURS_W.get(x, None)) if m == "ours" else (lambda x: x)
            w0, w1 = wv.weave0.map(wmap), wv.weave.map(wmap)
            ok_w = w0.notna() & w1.notna()
            r = dict(set=s, model=m, cond=cn, family=c["family"], kind=c["kind"], transform=c["fn"],
                     strength=c["strength"], n_images=h.image.nunique(), n_values=len(h),
                     n_values_total=len(all_rows), frac_in_band=100 * len(h) / max(1, len(all_rows)),
                     med_abs_drift_pct=st["med"], p90_abs_drift_pct=st["p90"], pct_gt5=st["gt5"],
                     pct_gt10=st["gross10"], median_signed_drift_pct=st["bias"],
                     cross_med_pct=stc["med"], cross_p90_pct=stc["p90"], cross_bias_pct=stc["bias"],
                     dens_med_pct=std_["med"], dens_p90_pct=std_["p90"], dens_bias_pct=std_["bias"],
                     floor_med_pct=f2["med"], floor_p90_pct=f2["p90"], floor_pct_gt10=f2["gross10"],
                     d_med_ci_lo=ci["med"][0], d_med_ci_hi=ci["med"][1], d_p90_ci_lo=ci["p90"][0],
                     d_p90_ci_hi=ci["p90"][1], d_gt10_ci_lo=ci["gross10"][0], d_gt10_ci_hi=ci["gross10"][1],
                     bias_ci_lo=ci["bias"][0], bias_ci_hi=ci["bias"][1],
                     sizefloor_med_pct=sz["med"] if sz else np.nan, sizefloor_p90_pct=sz["p90"] if sz else np.nan,
                     sizefloor_pct_gt10=sz["gross10"] if sz else np.nan,
                     weave_agree_pct=100 * float((w0[ok_w] == w1[ok_w]).mean()) if ok_w.any() else np.nan,
                     wilcoxon_n_images=len(jj), wilcoxon_p=wil_p,
                     PASS=bool(ok), PASS_sizematched=bool(ok_sz), PASS_conf=ok_conf,
                     conf_n_values=len(h[h.conf0 >= 0.5]) if m == "ours" else np.nan,
                     conf_med_pct=st_conf["med"] if st_conf else np.nan,
                     conf_p90_pct=st_conf["p90"] if st_conf else np.nan,
                     conf_pct_gt10=st_conf["gross10"] if st_conf else np.nan)
            if "lab_cross" in h and h["lab_cross"].notna().any():
                lab = np.where(m == "ours", h.lab_cross, h.lab_dens)
                lab0 = np.where(m == "ours", h.lab_cross0, h.lab_dens0)
                ape = np.abs(h.obs / lab - 1).fillna(1.0)
                ape0 = np.abs(h.base / lab0 - 1).fillna(1.0)
                r["exact_MAPE_pct"] = 100 * float(np.mean(ape))
                r["exact_MdAPE_pct"] = 100 * float(np.median(ape))
                r["exact_acc10_pct"] = 100 * float(np.mean(ape <= 0.10))
                r["exact_clean_MdAPE_pct"] = 100 * float(np.median(ape0))
                r["exact_clean_acc10_pct"] = 100 * float(np.mean(ape0 <= 0.10))
                # paired cluster bootstrap over images of the accuracy change vs the clean image
                dd = pd.DataFrame({"image": h.image.values, "a": (ape <= .1).astype(float).values,
                                   "a0": (ape0 <= .1).astype(float).values, "p": ape.values, "p0": ape0.values})
                gi = {k: g_[["a", "a0", "p", "p0"]].values for k, g_ in dd.groupby("image")}
                ks = np.array(list(gi))
                rb = np.random.default_rng(zlib.crc32((cn + m).encode()))
                da, dm = [], []
                for _ in range(400):
                    X = np.concatenate([gi[k] for k in rb.choice(ks, len(ks))])
                    da.append(100 * (X[:, 0].mean() - X[:, 1].mean()))
                    dm.append(100 * (np.median(X[:, 2]) - np.median(X[:, 3])))
                r["exact_d_acc10_pp"] = r["exact_acc10_pct"] - r["exact_clean_acc10_pct"]
                r["exact_d_acc10_ci_lo"], r["exact_d_acc10_ci_hi"] = np.percentile(da, [2.5, 97.5])
                r["exact_d_MdAPE_pp"] = r["exact_MdAPE_pct"] - r["exact_clean_MdAPE_pct"]
                r["exact_d_MdAPE_ci_lo"], r["exact_d_MdAPE_ci_hi"] = np.percentile(dm, [2.5, 97.5])
            rows.append(r)
    S = pd.DataFrame(rows)
    # Holm adjustment of the Wilcoxon p-values within each (set, model)
    S["wilcoxon_p_holm"] = np.nan
    for _, g in S.groupby(["set", "model"]):
        g = g[g.wilcoxon_p.notna()].sort_values("wilcoxon_p")
        m_ = len(g)
        adj = np.minimum(1.0, np.maximum.accumulate((m_ - np.arange(m_)) * g.wilcoxon_p.values))
        S.loc[g.index, "wilcoxon_p_holm"] = adj
    # exact-label clean reference on synth100
    for m in S.model.unique():
        g = e[(e.set == "synth100") & (e.model == m) & (e.cond == "clean") & e.in_band]
        if len(g) and "lab_cross" in g:
            lab = g.lab_cross if m == "ours" else g.lab_dens
            ape = np.abs(g.obs / lab - 1)
            S.loc[(S.set == "synth100") & (S.model == m), "clean_exact_MAPE_pct"] = 100 * float(np.mean(ape))
            S.loc[(S.set == "synth100") & (S.model == m), "clean_exact_acc10_pct"] = 100 * float(np.mean(ape <= .1))
    order = {c["cond"]: k for k, c in enumerate(conditions())}
    S["_o"] = S.cond.map(order)
    S = S.sort_values(["model", "set", "_o"]).drop(columns="_o")
    S.to_csv(os.path.join(DEST, "augment_verification.csv"), index=False, float_format="%.4g")
    print(f"wrote augment_verification.csv ({len(S)} rows)")
    return S, e


# =============================================================================================
# whitelist, diagnostics, figure
# =============================================================================================
SIZE_CHANGING = {"zoom_0.5", "zoom_0.7", "zoom_1.25", "zoom_1.6", "zoom_2", "zoomxy_1.25x1", "zoomxy_1x0.8",
                 "tile_0.5in", "tile_0.25in"}
# (policy key, condition, strength, direction); conditions in a direction are ordered mild -> strong
LADDERS = {
    "rotate": [("rot_inner_+2", 2, "+"), ("rot_inner_+5", 5, "+"), ("rot_inner_+10", 10, "+"), ("rot_inner_+15", 15, "+"),
               ("rot_inner_-2", 2, "-"), ("rot_inner_-5", 5, "-"), ("rot_inner_-10", 10, "-"), ("rot_inner_-15", 15, "-")],
    "rotate_reflect": [("rot_reflect_+5", 5, "+"), ("rot_reflect_+15", 15, "+")],
    "brightness": [("bright_0.6", 0.6, "-"), ("bright_1.4", 1.4, "+")],
    "contrast": [("contrast_0.6", 0.6, "-"), ("contrast_1.5", 1.5, "+")],
    "gamma": [("gamma_0.6", 0.6, "-"), ("gamma_1.7", 1.7, "+")],
    "hue_shift": [("hue_+60", 60, "+")],
    "saturation": [("sat_0.25", 0.25, "-"), ("sat_1.75", 1.75, "+")],
    "greyscale": [("greyscale", 1, "+")],
    "clahe": [("clahe_2", 2, "+"), ("clahe_4", 4, "+")],
    "white_balance": [("wb_warm", 1.25, "+"), ("wb_cool", 0.75, "-")],
    "gaussian_noise": [("noise_0.01", 0.01, "+"), ("noise_0.03", 0.03, "+"), ("noise_0.06", 0.06, "+")],
    "jpeg": [("jpeg_70", 70, "-"), ("jpeg_40", 40, "-"), ("jpeg_20", 20, "-")],
    "gaussian_blur_px": [("blur_0.5px", 0.5, "+"), ("blur_1px", 1.0, "+"), ("blur_1.5px", 1.5, "+")],
    "gaussian_blur": [("blur_0.1pitch", 0.1, "+"), ("blur_0.2pitch", 0.2, "+")],
    "motion_blur": [("motion_3px", 3, "+"), ("motion_7px", 7, "+")],
    "sharpen": [("sharpen_1", 1.0, "+"), ("sharpen_2.5", 2.5, "+")],
    "glare": [("glare_mild", 0.4, "+"), ("glare_strong", 1.0, "+")],
    "shadow": [("shadow_soft", 0.5, "-"), ("shadow_deep", 0.15, "-")],
    "vignette": [("vignette_0.3", 0.3, "+"), ("vignette_0.6", 0.6, "+")],
    "zoom": [("zoom_1.25", 1.25, "+"), ("zoom_1.6", 1.6, "+"), ("zoom_2", 2.0, "+"),
             ("zoom_0.7", 0.7, "-"), ("zoom_0.5", 0.5, "-")],
    "zoom_xy": [("zoomxy_1.25x1", 1.25, "+"), ("zoomxy_1x0.8", 0.8, "-")],
    "tile": [("tile_0.5in", 0.5, "+"), ("tile_0.25in", 0.25, "+")],
    "perspective": [("persp_0.01", 0.01, "+"), ("persp_0.02", 0.02, "+"), ("persp_0.03", 0.03, "+")],
    "hflip": [("hflip", 1, "+")], "vflip": [("vflip", 1, "+")],
    "rot90": [("rot90", 90, "+"), ("rot270", 270, "+")],
}


def verdict_table(S):
    """One row per condition with the verdicts that enter the whitelist."""
    out = []
    for cn, g in S.groupby("cond", sort=False):
        r = dict(cond=cn, family=g.family.iloc[0], kind=g.kind.iloc[0])
        for (s, m), h in g.groupby(["set", "model"]):
            h = h.iloc[0]
            use = "PASS_sizematched" if cn in SIZE_CHANGING else "PASS"
            r[f"{m}_{s}"] = bool(h[use])
            r[f"{m}_{s}_med"], r[f"{m}_{s}_p90"] = h.med_abs_drift_pct, h.p90_abs_drift_pct
            r[f"{m}_{s}_gt10"], r[f"{m}_{s}_bias"] = h.pct_gt10, h.median_signed_drift_pct
        r["label_preserving"] = bool(r.get("ours_real150", False) and r.get("ours_synth100", False))
        exact_map = (r["kind"] == "candidate" and not cn.startswith("rot_reflect"))
        unb = all(abs(r.get(f"{m}_{st}_bias", np.inf)) <= lim
                  for m, lim in (("ours", 1.0), ("cnn", 2.0)) for st in ("real150", "synth100"))
        r["unbiased_both"] = bool(unb)
        r["tier"] = "A" if r["label_preserving"] else ("B" if (exact_map and unb) else "C")
        out.append(r)
    return pd.DataFrame(out)


def make_whitelist(V):
    wl = _whitelist(V, "label_preserving")
    ext = _whitelist(V.assign(_ab=V.tier.isin(["A", "B"])), "_ab")
    wl["transforms_extended"] = ext["transforms"]
    wl["policy_extended"] = ext["policy"]
    wl["tiers"] = {r.cond: r.tier for r in V.itertuples() if r.kind in ("candidate", "null")}
    return wl


def _whitelist(V, flag):
    vv = V.set_index("cond")
    wl = {"rule": "label_preserving = within clean-run noise for ours on real150 AND synth100 (median |drift| <= "
                  "floor+0.5pp, p90 <= floor+2pp, P(>10%) <= floor+2pp, |median signed drift| <= 1%); floor = "
                  "pooled JPEG-q95 re-encode + 0.5-px shift; size-changing transforms use max(floor, size-matched "
                  "floor). legacy-13 drift is reported only and never enters this rule.",
          "transforms": {}, "policy": {}, "disabled": []}
    for key, ladder in LADDERS.items():
        res = {"tested": [], "safe": {}}
        for d in sorted({x[2] for x in ladder}):
            ok_upto = None
            for cn, lvl, dd in ladder:
                if dd != d or cn not in vv.index:
                    continue
                ok = bool(vv.loc[cn, flag])
                res["tested"].append({"cond": cn, "level": lvl, "passes_tier": ok,
                                      "cnn_real150_within_noise": bool(vv.loc[cn].get("cnn_real150", False))})
                if not ok:
                    break
                ok_upto = lvl
            res["safe"][d] = ok_upto
        wl["transforms"][key] = res
    T = wl["transforms"]

    def two(key, ident=1.0):
        lo, hi = T[key]["safe"].get("-"), T[key]["safe"].get("+")
        return [lo if lo is not None else ident, hi if hi is not None else ident]

    pol = {}
    for k in ("hflip", "vflip"):
        if T[k]["safe"].get("+"):
            pol[k] = {"p": 0.5}
    if T["rot90"]["safe"].get("+"):
        pol["rot90"] = {"p": 0.5}
    a, b = T["rotate"]["safe"].get("+"), T["rotate"]["safe"].get("-")
    if a and b:
        m = float(min(a, b))
        pol["rotate"] = {"p": 0.5, "deg": [-m, m], "mode": "inner"}
    for k, arg in (("brightness", "factor"), ("contrast", "factor"), ("gamma", "g"), ("saturation", "factor")):
        rng_ = two(k)
        if rng_ != [1.0, 1.0]:
            pol[k] = {"p": 0.5 if k in ("brightness", "contrast", "gamma") else 0.3, arg: rng_}
    if T["hue_shift"]["safe"].get("+"):
        h = float(T["hue_shift"]["safe"]["+"])
        pol["hue_shift"] = {"p": 0.3, "deg": [-h, h]}
    if T["greyscale"]["safe"].get("+"):
        pol["greyscale"] = {"p": 0.1}
    if T["clahe"]["safe"].get("+"):
        pol["clahe"] = {"p": 0.2, "clip": [1.0, float(T["clahe"]["safe"]["+"])]}
    wb = two("white_balance")
    if wb != [1.0, 1.0]:
        pol["white_balance"] = {"p": 0.3, "gain": [min(wb), max(wb)]}
    if T["gaussian_noise"]["safe"].get("+"):
        pol["gaussian_noise"] = {"p": 0.3, "sigma": [0.0, float(T["gaussian_noise"]["safe"]["+"])]}
    if T["jpeg"]["safe"].get("-"):
        pol["jpeg"] = {"p": 0.3, "q": [int(T["jpeg"]["safe"]["-"]), 95]}
    if T["gaussian_blur"]["safe"].get("+"):
        pol["gaussian_blur"] = {"p": 0.2, "sigma_pitch": [0.0, float(T["gaussian_blur"]["safe"]["+"])]}
    if T["gaussian_blur_px"]["safe"].get("+"):
        pol["gaussian_blur_px"] = {"p": 0.2, "sigma": [0.0, float(T["gaussian_blur_px"]["safe"]["+"])]}
    if T["motion_blur"]["safe"].get("+"):
        pol["motion_blur"] = {"p": 0.1, "length": [1, int(T["motion_blur"]["safe"]["+"])]}
    if T["sharpen"]["safe"].get("+"):
        pol["sharpen"] = {"p": 0.2, "amount": [0.0, float(T["sharpen"]["safe"]["+"])]}
    if T["glare"]["safe"].get("+"):
        st = T["glare"]["safe"]["+"]
        pol["glare"] = {"p": 0.2, "n_blobs": [1, 2 if st < 1 else 4], "amp": [0.2, float(st)]}
    if T["shadow"]["safe"].get("-"):
        pol["shadow"] = {"p": 0.3, "depth": [float(T["shadow"]["safe"]["-"]), 0.9]}
    if T["vignette"]["safe"].get("+"):
        pol["vignette"] = {"p": 0.3, "strength": [0.0, float(T["vignette"]["safe"]["+"])]}
    z = [T["zoom"]["safe"].get("-") or 1.0, T["zoom"]["safe"].get("+") or 1.0]
    if z != [1.0, 1.0]:
        # zoom-out keeps the label only while the yarns stay resolved: ours' gross-flip rate under zoom 0.5
        # is within the floor only at >= 6 output px per yarn (aug_verify/bins_px_per_thread.csv)
        pol["zoom"] = {"p": 0.5, "s": [float(z[0]), float(z[1])], "min_px_per_thread": 6.0}
    if T["perspective"]["safe"].get("+"):
        pol["perspective"] = {"p": 0.3, "eps": [0.0, float(T["perspective"]["safe"]["+"])]}
    wl["policy"] = pol
    wl["disabled"] = sorted(set(A.DEFAULT_POLICY) - set(pol))
    wl["excluded"] = A.EXCLUDED
    return wl


def ppt_bins(raw, e):
    """Drift of zoom-out / px-blur conditions by output pixels per thread (ours and CNN, real150+synth100)."""
    side = raw[raw.cond == "clean"][["set", "image", "side"]].drop_duplicates(["set", "image"])
    cl = e[(e.cond == "clean") & (e.model == "ours")].groupby(["set", "image"]).base.max().rename("cmax").reset_index()
    out = []
    for cn, s, sig in (("clean", 1.0, 0.0), ("null_jpeg95", 1.0, 0.0), ("zoom_0.7", 0.7, 0.0), ("zoom_0.5", 0.5, 0.0),
                       ("blur_0.5px", 1.0, 0.5), ("blur_1px", 1.0, 1.0), ("blur_1.5px", 1.0, 1.5)):
        h = e[(e.cond == cn) & e.in_band & e.set.isin(["real150", "synth100"])]
        h = h.merge(side, on=["set", "image"]).merge(cl, on=["set", "image"])
        h["ppt_out"] = np.minimum(h.side, 1024) * s / h.cmax
        h["sigma_over_pitch"] = sig / (np.minimum(h.side, 1024) / h.cmax)
        h["bin"] = pd.cut(h.ppt_out, [0, 3, 4, 6, 1e9], labels=["<3", "3-4", "4-6", ">=6"])
        for (m, b), hh in h.groupby(["model", "bin"], observed=True):
            st = stats(hh.primary.values)
            out.append(dict(cond=cn, model=m, ppt_out_bin=str(b), n_values=len(hh), med_abs_drift_pct=st["med"],
                            p90_abs_drift_pct=st["p90"], pct_gt10=st["gross10"], median_signed_drift_pct=st["bias"],
                            median_sigma_over_pitch=float(np.median(hh.sigma_over_pitch))))
    return pd.DataFrame(out)


def tilt_bins(e):
    """Rotation drift binned by the predicted crossing-count change |fc - 1| (i.e. by the total tilt
    |alpha -/+ theta|; ours' lattice searches +-15 deg), with drift against both conventions."""
    out = []
    for cn in [c for c in COND if c.startswith("rot_inner_") or c.startswith("rot_reflect_")]:
        h = e[(e.cond == cn) & e.in_band & e.set.isin(["real150", "synth100"])].copy()
        if not len(h):
            continue
        h["bin"] = pd.cut(np.abs(h.fc - 1) * 100, [-1, 1, 3, 100], labels=["cos-loss<=1%", "1-3%", ">3%"])
        for (m, b), hh in h.groupby(["model", "bin"], observed=True):
            st = stats(hh.primary.values)
            sd = stats(hh.drift_dens.values)
            out.append(dict(cond=cn, model=m, crossing_change_bin=str(b), n_values=len(hh),
                            med_abs_drift_pct=st["med"], p90_abs_drift_pct=st["p90"], pct_gt10=st["gross10"],
                            median_signed_drift_cross_pct=stats(hh.drift_cross.values)["bias"],
                            median_signed_drift_dens_pct=sd["bias"]))
    return pd.DataFrame(out)


FIG_ROWS_SKIP = ("clean",)
NICE_FAMILY = {"null": "null (noise floor)", "a_flip": "(a) flips", "b_rot90": "(b) rot90", "c_rotate": "(c) small rotation",
               "d_photometric": "(d) photometric", "e_sensor": "(e) sensor / compression",
               "f_illumination": "(f) illumination", "g_zoom": "(g) zoom (physical)", "h_tile": "(h) tiles (physical)",
               "i_perspective": "(i) perspective", "x_excluded": "excluded (negative controls)"}


def summary_table(S, V):
    rows = []
    for cn in [c["cond"] for c in conditions() if c["kind"] not in ("clean", "sizefloor")]:
        g = S[S.cond == cn]
        if not len(g):
            continue
        r = {"family": COND[cn]["family"], "cond": cn, "kind": COND[cn]["kind"]}
        for (m, st) in (("ours", "real150"), ("ours", "synth100"), ("cnn", "real150"), ("cnn", "synth100"),
                        ("ours", "legacy13"), ("cnn", "legacy13")):
            h = g[(g.model == m) & (g.set == st)]
            if not len(h):
                continue
            h = h.iloc[0]
            pre = f"{m}_{st}"
            r[f"{pre}_n"] = int(h.n_values)
            r[f"{pre}_med"] = h.med_abs_drift_pct
            r[f"{pre}_p90"] = h.p90_abs_drift_pct
            r[f"{pre}_gt10"] = h.pct_gt10
            r[f"{pre}_bias"] = h.median_signed_drift_pct
            r[f"{pre}_pass"] = bool(h.PASS_sizematched if cn in SIZE_CHANGING else h.PASS)
            if m == "ours":
                r[f"{pre}_pass_conf"] = h.PASS_conf
            if st == "synth100" and "exact_MAPE_pct" in h:
                r[f"{pre}_exactMAPE"] = h.exact_MAPE_pct
                r[f"{pre}_exactAcc10"] = h.exact_acc10_pct
                r[f"{pre}_clean_exactMAPE"] = h.get("clean_exact_MAPE_pct", np.nan)
        vv = V[V.cond == cn]
        r["label_preserving"] = bool(vv.label_preserving.iloc[0]) if len(vv) else False
        r["tier"] = vv.tier.iloc[0] if len(vv) else ""
        rows.append(r)
    return pd.DataFrame(rows)


def latex_table(T, path):
    lines = [r"\begin{tabular}{llrrrrlrrl}", r"\toprule",
             r"family & transform & \multicolumn{4}{c}{ours, real-150: $|$drift$|$ (\%)} & & "
             r"\multicolumn{2}{c}{CNN, real-150} & \\",
             r" & & median & p90 & $>$10\% & bias & synth & median & p90 & verdict \\", r"\midrule"]
    fam = None
    for r in T.itertuples():
        f = NICE_FAMILY.get(r.family, r.family) if r.family != fam else ""
        fam = r.family
        verdict = {"A": "A (verified)", "B": "B (exact, unbiased)"}.get(r.tier, "C") if r.kind == "candidate" \
            else ("excluded" if r.kind == "excluded" else "floor")
        g = lambda k: getattr(r, k, np.nan)  # noqa: E731
        lines.append(f"{f} & {r.cond.replace('_', ' ')} & {g('ours_real150_med'):.2f} & {g('ours_real150_p90'):.1f} & "
                     f"{g('ours_real150_gt10'):.1f} & {g('ours_real150_bias'):+.2f} & "
                     f"{'pass' if g('ours_synth100_pass') is True else 'fail'} & {g('cnn_real150_med'):.2f} & "
                     f"{g('cnn_real150_p90'):.1f} & {verdict} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(path, "w").write("\n".join(lines) + "\n")


def make_figure(S, path_base, V=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    BLUE, ORANGE, GREY, INK, INK2, GRID = "#2a78d6", "#eb6834", "#8a8984", "#0b0b0b", "#52514e", "#e5e4e0"
    conds = [c for c in conditions() if c["kind"] not in ("clean", "sizefloor")]
    panels = [("ours", "real150", "ours (frozen classical), real-150"),
              ("cnn", "real150", "synthetic-only CNN, real-150"),
              ("ours", "synth100", "ours, synthetic-100")]
    n = len(conds)
    H = 0.135 * n + 1.8
    fig, axes = plt.subplots(1, 3, figsize=(10.5, H), sharey=True, gridspec_kw=dict(wspace=0.2))
    ypos, fam_rows, y = {}, [], 0
    last = None
    for c in conds:
        if c["family"] != last:
            if last is not None:
                y += 0.6
            fam_rows.append((y, c["family"]))
            last = c["family"]
        ypos[c["cond"]] = y
        y += 1
    for ax, (m, st, title) in zip(axes, panels):
        g = S[(S.model == m) & (S.set == st)].set_index("cond")
        if not len(g):
            continue
        fl = g[g.kind != "null"]                        # every non-null row carries the pooled floor
        p90_thr = float(fl.floor_p90_pct.median() + 2.0)
        med_thr = float(fl.floor_med_pct.median() + 0.5)
        ax.axvspan(0.01, p90_thr, color="#f0efec", zorder=0, lw=0)
        ax.axvline(p90_thr, color=INK2, lw=0.6, ls=(0, (3, 2)), zorder=1)
        ax.axvline(med_thr, color=INK2, lw=0.6, ls=(0, (1, 2)), zorder=1)
        for c in conds:
            cn = c["cond"]
            if cn not in g.index:
                continue
            r = g.loc[cn]
            yy = ypos[cn]
            ok = bool(r.PASS_sizematched if cn in SIZE_CHANGING else r.PASS)
            lo, hi = max(0.01, r.med_abs_drift_pct), max(0.01, r.p90_abs_drift_pct)
            if c["kind"] == "excluded":
                col, mk, fc = GREY, "X", GREY
            elif ok:
                col, mk, fc = BLUE, "o", BLUE
            else:
                col, mk, fc = ORANGE, "D", "white"
            ax.plot([lo, hi], [yy, yy], color=col, lw=1.2, solid_capstyle="round", zorder=2)
            ax.plot([hi], [yy], marker="|", color=col, ms=5, mew=1.2, zorder=2)
            ax.plot([lo], [yy], marker=mk, color=col, mfc=fc, ms=4.2, mew=1.0, zorder=3)
            if r.pct_gt10 > 0:
                ax.text(1.015, yy, f"{r.pct_gt10:.0f}", fontsize=5.5, color=INK2, va="center", ha="left",
                        transform=ax.get_yaxis_transform())
        ax.set_xscale("log")
        ax.set_xlim(0.01, 400)
        ax.set_xticks([0.01, 0.1, 1, 10, 100])
        ax.set_xticklabels(["0.01", "0.1", "1", "10", "100"], fontsize=6.5, color=INK2)
        ax.set_xlabel("|drift| (%)", fontsize=6.5, color=INK2)
        ax.set_title(title, fontsize=8, color=INK, loc="left")
        ax.grid(axis="x", color=GRID, lw=0.5, zorder=0)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(axis="y", length=0)
        ax.text(1.015, -1.3, ">10%", fontsize=5.5, color=INK2, ha="left", va="center",
                transform=ax.get_yaxis_transform())
    axes[0].set_yticks([ypos[c["cond"]] for c in conds])
    tiers = {} if V is None else dict(zip(V.cond, V.tier))
    axes[0].set_yticklabels([c["cond"] + (f"  [{tiers[c['cond']]}]" if c["kind"] == "candidate" and c["cond"] in tiers
                                          else "") for c in conds], fontsize=6, color=INK)
    for yy, fam in fam_rows:
        axes[0].annotate(NICE_FAMILY.get(fam, fam), xy=(0, yy - 0.75), xycoords=("axes fraction", "data"),
                         xytext=(-118, 0), textcoords="offset points", fontsize=6.5, color=INK2,
                         fontweight="bold", va="center", ha="left", annotation_clip=False)
    axes[0].set_ylim(y, -2)
    leg = [Line2D([], [], color=BLUE, marker="o", lw=1.2, ms=4, label="within clean-run noise (PASS)"),
           Line2D([], [], color=ORANGE, marker="D", mfc="white", lw=1.2, ms=4, label="outside noise (FAIL)"),
           Line2D([], [], color=GREY, marker="X", lw=1.2, ms=4, label="excluded transform (negative control)"),
           Line2D([], [], color=INK2, lw=0.6, ls=(0, (1, 2)), label="median tolerance (floor + 0.5 pp)"),
           Line2D([], [], color=INK2, lw=0.6, ls=(0, (3, 2)), label="p90 tolerance (floor + 2 pp)")]
    fig.legend(handles=leg, loc="upper center", ncol=5, fontsize=6.5, frameon=False, bbox_to_anchor=(0.55, 1.0))
    fig.text(0.19, 0.08 / H, "drift = observed / predicted label change - 1, per image and output axis (h<->v swapped "
             "where the transform swaps them);\ndot = median |drift|, bar end = 90th percentile; right column = % of "
             "values with |drift| > 10 %; shaded = p90 tolerance of the pass rule.\n[A] within clean-run noise for ours "
             "on real-150 and synth-100; [B] label map exact and unbiased on both instruments (ours <= 1 %, CNN <= 2 %); "
             "[C] neither.", fontsize=6, color=INK2)
    fig.subplots_adjust(left=0.19, right=0.975, top=1 - 0.55 / H, bottom=0.95 / H)
    for ext in ("pdf", "png"):
        fig.savefig(f"{path_base}.{ext}", dpi=200)
    plt.close(fig)


def report():
    S, e = summarise()
    raw = pd.concat([pd.read_csv(f, dtype={"image": str}) for f in sorted(__import__("glob").glob(
        os.path.join(OUT, "raw_*.csv")))], ignore_index=True)
    V = verdict_table(S)
    T = summary_table(S, V)
    T.to_csv(os.path.join(DEST, "augment_verification_summary.csv"), index=False, float_format="%.4g")
    latex_table(T, os.path.join(DEST, "augment_verification_table.tex"))
    wl = make_whitelist(V)
    # attach the evidence behind every tested level (data only; the tiers are decided by the rules above)
    idx = S.set_index(["set", "model", "cond"])
    for key in ("transforms", "transforms_extended"):
        for tr in wl[key].values():
            for t in tr["tested"]:
                cn = t["cond"]
                ev = {}
                for st, m in (("real150", "ours"), ("synth100", "ours"), ("real150", "cnn"), ("synth100", "cnn")):
                    if (st, m, cn) in idx.index:
                        r = idx.loc[(st, m, cn)]
                        ev[f"{m}_{st}"] = {k: round(float(r[k]), 3) for k in (
                            "med_abs_drift_pct", "p90_abs_drift_pct", "pct_gt10", "median_signed_drift_pct")}
                        if st == "synth100" and "exact_d_acc10_pp" in r and np.isfinite(r["exact_d_acc10_pp"]):
                            ev[f"{m}_{st}"]["exact_label_d_acc10_pp"] = [round(float(r[k]), 2) for k in (
                                "exact_d_acc10_pp", "exact_d_acc10_ci_lo", "exact_d_acc10_ci_hi")]
                t["evidence"] = ev
                t["tier"] = wl["tiers"].get(cn)
    wl["notes"] = [
        "policy (tier A) is the default of augment.load_policy(); policy_extended adds tier B.",
        "Tier B rotations are unbiased (ours median signed drift <= 0.16 %) but both instruments lose accuracy "
        "vs the exactly transformed synthetic label as the angle grows (see evidence.*.exact_label_d_acc10_pp); "
        "at +-5 deg the CNN's change includes 0, at +-10 deg it does not. Start training at +-5 deg.",
        "Zoom-out keeps the label only while yarns stay resolved: require >= 6 output px per yarn "
        "(policy_extended.zoom.min_px_per_thread; aug_verify/bins_px_per_thread.csv).",
        "Physical tiles are tier C only because ours reads 0.5-inch tiles 1.5 % low on real crops (synthetic: "
        "-0.3 %, CNN: +0.1 %); the label map itself is exact (test_augment.py).",
        "The existing synthetic-only CNN is NOT invariant to several exactly label-preserving transforms "
        "(rot90: median per-axis change 5.8 %, contrast 0.6: +4.5 % bias on real-150), which is the case for "
        "training with them.",
        "legacy-13 drift is in augment_verification.csv for transparency and was never used for any decision.",
    ]
    wl["n"] = {f"{m}_{st}": int(S[(S.model == m) & (S.set == st)].n_images.max())
               for m, st in S[["model", "set"]].drop_duplicates().itertuples(index=False)}
    with open(os.path.join(DEST, "augment_whitelist.json"), "w") as f:
        json.dump(wl, f, indent=1, default=lambda o: o if not isinstance(o, np.generic) else o.item())
    pb = ppt_bins(raw, e)
    pb.to_csv(os.path.join(DEST if DEST != HERE else OUT, "bins_px_per_thread.csv"), index=False, float_format="%.4g")
    tb = tilt_bins(e)
    tb.to_csv(os.path.join(DEST if DEST != HERE else OUT, "bins_rotation_tilt.csv"), index=False, float_format="%.4g")
    figdir = os.path.join(DEST, "figs") if DEST == HERE else DEST
    os.makedirs(figdir, exist_ok=True)
    make_figure(S, os.path.join(figdir, "augment_invariance"), V)
    print("label-preserving:", ", ".join(V[V.label_preserving].cond))
    print("policy:", json.dumps(wl["policy"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["ours", "cnn", "summarise", "report"])
    ap.add_argument("--sets", default="real150,synth100,legacy13")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--conds", default=None, help="comma list (default: all)")
    a = ap.parse_args()
    sets = tuple(a.sets.split(","))
    conds = a.conds.split(",") if a.conds else None
    if a.stage == "ours":
        run_ours(sets, a.workers, conds)
    elif a.stage == "cnn":
        run_cnn(sets, conds)
    elif a.stage == "summarise":
        summarise()
    else:
        report()


if __name__ == "__main__":
    main()
