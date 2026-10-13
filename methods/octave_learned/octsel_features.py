"""octsel_features.py -- LABEL-FREE per-file, per-axis features for the learned octave selector octsel-v1.

Reads no human label.  For every one of the 150 files (methods/predictions_ours_150.csv) and each axis
ax in {h, v}, with c = the frozen counter's final count on that axis (ax_count):

  lr_lattice, lr_fft, lr_profile, lr_tiles, lr_gabor : log2(stage estimate / c)   (missing -> 0)
  lr_alt      : log2(alt_ax / c)  (the counter's alternative count; missing -> 0)
  g_c, g_2c, g_half : whitened spectral evidence (Spectrum.lookup 'g', the lattice score's unit) of the frozen
                counter at its counted frequency c, at 2c and at c/2, along the counter's own yarn direction
                (the lattice's centroid vector rescaled so that its axis component equals c; on-axis if the
                lattice found nothing).  Outside the counter's band / below 5 cycles -> 0 (no evidence).
  conf        : the counter's confidence (per file, same on both axes)
  lpx         : log2(S / c) = log2 px per yarn at the counted frequency (S = rect size)
  lr_gpt      : log2(GPT-6.1-sol P1 / c); GPT axis convention of score150.py: our h = gpt-6.1-sol__v,
                our v = gpt-6.1-sol__h (missing -> 0, flag gpt_missing)

The spectral part re-runs the FROZEN counter's label-free code path (preprocess -> Spectrum -> est_lattice),
imported read-only from methods/results/counting_frozen_final.py (md5 checked; no bytecode written), on the
frozen rectified crop (common/rectify.py), exactly as handover/.../analysis_A2/a2_spectral.py does.

    .tools/venv/bin/python -I -B methods/octave_learned/octsel_features.py [--workers 4]
writes methods/octave_learned/features_octsel_150.csv (one row per file and axis).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import math
import os
import sys

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
FROZEN = os.path.join(ROOT, "methods", "results", "counting_frozen_final.py")
FROZEN_MD5 = "3d0520d6bec8a016543ce7e5b353d784"
P_OURS = os.path.join(ROOT, "methods", "predictions_ours_150.csv")
P_VLM = os.path.join(ROOT, "vlm", "results", "rect150_P1_predictions_wide.csv")
OUT = os.path.join(HERE, "features_octsel_150.csv")
FEATURES = ["lr_lattice", "lr_fft", "lr_profile", "lr_tiles", "lr_gabor", "lr_alt",
            "g_c", "g_2c", "g_half", "conf", "lpx", "lr_gpt"]
_C = None
_R = None


def frozen():
    global _C, _R
    if _C is None:
        spec = importlib.util.spec_from_file_location("counting_frozen_final_ro", FROZEN)
        _C = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_C)
        sys.path.insert(0, os.path.join(ROOT, "common"))
        import rectify  # noqa: E402
        _R = rectify
    return _C, _R


def _g(sp, u, v):
    g, r = sp.lookup([u], [v])
    lo, hi = sp.band
    if not np.isfinite(g[0]) or r[0] > hi * 1.02 or r[0] < 5:
        return np.nan
    return float(g[0])


def spectral(args):
    """args = (file id, final h count, final v count) -> dict of g at c, 2c, c/2 per axis."""
    i, ch, cv = args
    C, R = frozen()
    crop = R.rectify(int(i))
    g, w, N, info = C.preprocess(crop)
    sp = C.Spectrum(g)
    lat = C.est_lattice(sp)
    out = {"id": int(i), "N_work": int(N), "lattice_found": lat is not None}
    cnt = {"h": ch, "v": cv}
    for ax, col, k in (("h", 0, 0), ("v", 1, 1)):
        c = float(cnt[ax])
        if lat is not None:
            vec = np.array(lat["B"])[:, col].astype(float).copy()
            if vec[k] < 0:
                vec = -vec
            out[f"{ax}_lattice_rerun"] = float(abs(vec[k]))
            vec = vec * (c / abs(vec[k])) if abs(vec[k]) > 0 else (np.array([c, 0.0]) if k == 0 else np.array([0.0, c]))
        else:
            out[f"{ax}_lattice_rerun"] = np.nan
            vec = np.array([c, 0.0]) if k == 0 else np.array([0.0, c])
        out[f"{ax}_g_c"] = _g(sp, *vec)
        out[f"{ax}_g_2c"] = _g(sp, *(2.0 * vec))
        out[f"{ax}_g_half"] = _g(sp, *(0.5 * vec))
        out[f"{ax}_angle_deg"] = float(math.degrees(math.atan2(abs(vec[1 - k]), abs(vec[k]))))
    return out


def _lr(a, c):
    a = pd.to_numeric(a, errors="coerce")
    r = np.log2(a / c)
    return r.where(np.isfinite(r) & (a > 0), 0.0)


def build(workers=4, p_ours=P_OURS, p_vlm=P_VLM):
    md5 = hashlib.md5(open(FROZEN, "rb").read()).hexdigest()
    if md5 != FROZEN_MD5:
        raise SystemExit(f"{FROZEN} md5 {md5} != frozen {FROZEN_MD5}")
    f = pd.read_csv(p_ours)
    w = pd.read_csv(p_vlm).set_index("image_id")
    items = [(int(r.id), float(r.h_count), float(r.v_count)) for r in f.itertuples()]
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(k, "1")
    if workers <= 1:
        sp = [spectral(a) for a in items]
    else:
        from multiprocessing import get_context
        with get_context("spawn").Pool(workers) as p:
            sp = p.map(spectral, items, chunksize=1)
    sp = pd.DataFrame(sp).set_index("id")
    rows = []
    for ax, gk in (("h", "gpt-6.1-sol__v"), ("v", "gpt-6.1-sol__h")):
        c = f[f"{ax}_count"].astype(float)
        d = pd.DataFrame({"id": f.id.astype(int), "axis": ax, "count": c, "S": f.S.astype(float)})
        for st in ("lattice", "fft", "profile", "tiles", "gabor"):
            d[f"lr_{st}"] = _lr(f[f"{ax}_{st}"], c).to_numpy()
        d["lr_alt"] = _lr(f[f"alt_{ax}"], c).to_numpy()
        s = sp.reindex(d.id)
        for gname in ("g_c", "g_2c", "g_half"):
            d[gname] = s[f"{ax}_{gname}"].to_numpy()
        d["conf"] = f.confidence.astype(float).fillna(0.0).to_numpy()
        d["lpx"] = np.log2(f.S.astype(float) / c).to_numpy()
        gpt = pd.to_numeric(w[gk], errors="coerce").reindex(f.id).to_numpy()
        d["gpt"] = gpt
        d["gpt_missing"] = ~(np.isfinite(gpt) & (gpt > 0))
        d["lr_gpt"] = _lr(pd.Series(gpt), c.reset_index(drop=True)).to_numpy()
        d["lattice_found"] = s["lattice_found"].to_numpy()
        d["lattice_rerun"] = s[f"{ax}_lattice_rerun"].to_numpy()
        d["lattice_stored"] = pd.to_numeric(f[f"{ax}_lattice"], errors="coerce").to_numpy()
        d["angle_deg"] = s[f"{ax}_angle_deg"].to_numpy()
        rows.append(d)
    out = pd.concat(rows, ignore_index=True)
    for gname in ("g_c", "g_2c", "g_half"):
        out[gname + "_raw"] = out[gname]
        out[gname] = out[gname].fillna(0.0)
    out = out.sort_values(["id", "axis"]).reset_index(drop=True)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--ours", default=P_OURS, help="frozen counter predictions (predictions_ours_150.csv format)")
    ap.add_argument("--gpt", default=P_VLM, help="VLM wide predictions (rect150_P1_predictions_wide.csv format)")
    a = ap.parse_args(argv)
    out = build(a.workers, a.ours, a.gpt)
    out.to_csv(a.out, index=False)
    chk = out.dropna(subset=["lattice_rerun", "lattice_stored"])
    rel = (chk.lattice_rerun / chk.lattice_stored - 1).abs()
    print(f"wrote {a.out}: {len(out)} rows; lattice re-run vs stored: {int((rel < 0.01).sum())}/{len(chk)} within 1% "
          f"(max rel {rel.max():.2e}); lattice found on {int(out.lattice_found.sum() // 2)} files; "
          f"GPT missing on {int(out.gpt_missing.sum())} axes")


if __name__ == "__main__":
    main()
