#!/usr/bin/env python
"""Evaluate a thread-density CNN checkpoint.

  * synthetic held-out set  (exact labels; breakdown by capture factor)
  * the 13 legacy hand-counted images (hand-count spreadsheet), zero-shot
      - axis-matched : warp -> h_count (vertical yarns), weft -> v_count
      - axis-swapped : warp -> v_count, weft -> h_count
      - best-assign. : per image, the permutation with the smaller error
        (orientation of the legacy photos is not recorded)
  * optional: predictions for the 150 unlabeled dataset images (rectified
    with the annotated corners) so they can be scored once counts arrive.

    python evaluate.py --ckpt runs/synth_r18/model.pt --synth-val data/synth_val \
        --legacy --real150 --out runs/synth_r18/eval
"""
from __future__ import annotations

import argparse
import json
import os
import time

import cv2
import numpy as np
import pandas as pd
import torch

import cnn_common as C
import fabric_data as fd


# ---------------------------------------------------------------------------
def load_synth_df(d):
    df = pd.read_csv(os.path.join(d, "labels.csv"))
    df["path"] = [os.path.join(d, f) for f in df["file"]]
    return df


def synth_items(d):
    df = load_synth_df(d)
    return [C.Item(None, r.path, float(r.h_count), float(r.v_count), C.weave_to_idx(r.weave_coarse))
            for r in df.itertuples()], df


RUN_PIPELINE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "methods", "legacy_2019", "run_pipeline.py")
METHODS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "methods")
LEGACY_MODES = ("detect", "raw", "trim", "vlm")
VLM_LEGACY_CROPS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vlm", "crops", "legacy")
TOOL_CROPS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "annotation_tool", "data", "crops")


def load_legacy_gt():
    """The 13 usable legacy hand counts.

    Uses the reference parser ``load_gt`` of the legacy reconstruction harness
    (run_pipeline.py) so every method in the paper scores against the same GT;
    falls back to the equivalent fabric_data parser if the harness is absent
    (both were checked to agree on all 13 images)."""
    if os.path.exists(RUN_PIPELINE):
        import importlib.util
        spec = importlib.util.spec_from_file_location("run_pipeline", RUN_PIPELINE)
        rp = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rp)
        gt, _ = rp.load_gt(fd.LEGACY_XLSX)
        out = []
        for k, v in gt.items():
            if not v["usable"]:
                continue
            wv = {"1/1": "plain", "2/1": "twill21", "1/2": "twill21"}.get(v["weave"], v["weave"] or None)
            rng = v["warp_lo"] != v["warp_hi"] or v["weft_lo"] != v["weft_hi"]
            out.append(fd.LegacyGT(name=k, file=fd.legacy_file(k), weft=v["weft"], warp=v["warp"], weave=wv,
                                   unclear=v["unclear"], is_range=bool(rng), raw=(v["warp_raw"], v["weft_raw"])))
        return out
    return fd.load_legacy_gt()


def legacy_crop(g, mode="detect"):
    """Square BGR crop of the 1-inch window of a legacy image.

    detect : frame detector shared with the classical counter
             (methods/legacy.py: Theil-Sen edge lines -> homography)
    raw    : the whole hand-cropped image is taken as the 1-inch window
    trim   : fabric_data.legacy_inner_window (per-side brightness trim)
    vlm    : the canonical paper crop vlm/crops/legacy/{name}.png (vlm/data_vlm.py frame trim,
             resized to a square), i.e. the exact pixels the VLMs / CountGD / Molmo2 saw"""
    if mode == "vlm":
        return C.cap_side(cv2.imread(os.path.join(VLM_LEGACY_CROPS, f"{g.name}.png")), 1024)
    im = cv2.imread(g.file)
    if mode == "raw":
        t = im
    elif mode == "trim":
        t = fd.legacy_inner_window(im)
    elif mode == "detect":
        import sys
        if METHODS_DIR not in sys.path:
            sys.path.insert(0, METHODS_DIR)
        import legacy as L
        corners, _ = L.detect_window(im)
        t = L.rectify_quad(im, corners)
    else:
        raise ValueError(mode)
    return C.cap_side(t, 1024)


def legacy_crops(mode="detect"):
    """[(LegacyGT, square BGR crop)]"""
    return [(g, legacy_crop(g, mode)) for g in load_legacy_gt()]


def _bins(x, edges, labels):
    return pd.cut(x, edges, labels=labels, include_lowest=True)


@torch.no_grad()
def eval_synth(model, d, device, n=None, Ps=(768,), flips=False, log=print):
    df = load_synth_df(d)
    if n:
        df = df.iloc[:n].copy()
    t0 = time.time()
    ph, pv, pw = [], [], []
    for k, r in enumerate(df.itertuples()):
        crop = C.cap_side(cv2.imread(r.path), 1024)
        o = C.predict_crop(model, crop, device, Ps=Ps, flips=flips)
        ph.append(o["h"]), pv.append(o["v"]), pw.append(int(np.argmax(o["weave_probs"])))
        if (k + 1) % 100 == 0:
            log(f"    synth eval {k + 1}/{len(df)}  {time.time() - t0:.0f}s")
    df["pred_h"], df["pred_v"], df["pred_weave"] = ph, pv, [C.WEAVE_CLASSES[i] for i in pw]
    df["ape_h"] = 100 * (df.pred_h - df.h_count).abs() / df.h_count
    df["ape_v"] = 100 * (df.pred_v - df.v_count).abs() / df.v_count
    df["ape"] = (df.ape_h + df.ape_v) / 2
    m = C.count_metrics(df.pred_h, df.pred_v, df.h_count, df.v_count)
    m["weave_acc"] = float(100 * (df.pred_weave == df.weave_coarse).mean())
    # capture-factor breakdown (mean APE over both axes, %)
    df["px_per_thread"] = df.r_eff / df[["h_count", "v_count"]].max(1)
    df["density_bin"] = _bins(df[["h_count", "v_count"]].max(1), [0, 60, 100, 140, 1e9],
                              ["<60", "60-100", "100-140", ">140"])
    df["ppt_bin"] = _bins(df.px_per_thread, [0, 3.5, 5, 8, 1e9], ["<3.5", "3.5-5", "5-8", ">8"])
    df["blur_bin"] = _bins(df.blur_sigma_px, [0, 0.7, 1.1, 10], ["<0.7", "0.7-1.1", ">1.1"])
    df["jpeg_bin"] = _bins(df.jpeg_q, [0, 60, 80, 101], ["<60", "60-80", ">80"])
    df["glare_on"] = df.glare > 0
    df["shadow_on"] = df.shadow > 0
    df["motion_on"] = df.motion_px > 0
    df["rot_bin"] = _bins(df.rotation_deg.abs(), [0, 3, 6, 90], ["<3", "3-6", ">6"])
    br = {}
    for col in ["weave_coarse", "density_bin", "ppt_bin", "blur_bin", "jpeg_bin", "glare_on",
                "shadow_on", "motion_on", "rot_bin", "colour_style"]:
        g = df.groupby(col, observed=True)["ape"].agg(["mean", "median", "count"])
        br[col] = {str(k): {"MAPE": round(float(v["mean"]), 2), "MdAPE": round(float(v["median"]), 2),
                            "n": int(v["count"])} for k, v in g.iterrows()}
    return df, m, br


@torch.no_grad()
def eval_legacy(model, device, Ps=C.P_TEST, flips=True, mode="detect"):
    rows = []
    for g, crop in legacy_crops(mode):
        o = C.predict_crop(model, crop, device, Ps=Ps, flips=flips)
        rows.append(dict(name=g.name, warp=g.warp, weft=g.weft, weave=g.weave, unclear=g.unclear,
                         is_range=g.is_range, pred_h=o["h"], pred_v=o["v"], h_iqr=o["h_iqr"],
                         v_iqr=o["v_iqr"], pred_weave=C.WEAVE_CLASSES[int(np.argmax(o["weave_probs"]))],
                         p_plain=float(o["weave_probs"][0]), p_twill=float(o["weave_probs"][1])))
    df = pd.DataFrame(rows)
    res = {}
    for subset, mask in [("all13", np.ones(len(df), bool)),
                         ("clear_no_range", (~df.unclear & ~df.is_range).values)]:
        s = df[mask]
        res[subset] = {
            "axis_matched": C.count_metrics(s.pred_h, s.pred_v, s.warp, s.weft),
            "axis_swapped": C.count_metrics(s.pred_v, s.pred_h, s.warp, s.weft),
        }
        pa, pb, sw = C.best_assignment(s.pred_h, s.pred_v, s.warp, s.weft)
        res[subset]["best_assignment"] = C.count_metrics(pa, pb, s.warp, s.weft)
        res[subset]["best_assignment"]["n_swapped"] = int(sw.sum())
        # sum of the two counts ("thread count") is orientation-free
        tot_p, tot_g = s.pred_h + s.pred_v, s.warp + s.weft
        res[subset]["total_thread_count"] = {
            "MAE": float((tot_p - tot_g).abs().mean()),
            "MAPE": float(100 * ((tot_p - tot_g).abs() / tot_g).mean())}
        gw = s.weave.map({"plain": "plain", "twill21": "twill"})
        res[subset]["weave_acc"] = float(100 * (s.pred_weave == gw).mean())
    # anisotropy diagnostics (n=13): does the model get the warp/weft ratio?
    lr_t, lr_p = np.log(df.warp / df.weft), np.log(df.pred_h / df.pred_v)
    res["logratio_corr_axis_matched"] = float(np.corrcoef(lr_t, lr_p)[0, 1])
    res["logratio_std_true"], res["logratio_std_pred"] = float(lr_t.std()), float(lr_p.std())
    gm_t, gm_p = np.sqrt(df.warp * df.weft), np.sqrt(df.pred_h * df.pred_v)
    res["geomean_density"] = {"MAE": float((gm_p - gm_t).abs().mean()),
                              "MAPE": float(100 * ((gm_p - gm_t).abs() / gm_t).mean()),
                              "n10": int(((gm_p - gm_t).abs() / gm_t <= 0.10).sum())}
    # best any ISOTROPIC predictor could do (pred_h = pred_v = true geometric mean)
    res["isotropic_oracle"] = C.count_metrics(gm_t, gm_t, df.warp, df.weft)
    # naive reference (oracle-ish: uses the mean of the 26 test counts themselves)
    mu = float(np.mean(np.concatenate([df.warp, df.weft])))
    res["reference_constant_mean"] = C.count_metrics(np.full(len(df), mu), np.full(len(df), mu),
                                                     df.warp, df.weft)
    return df, res


@torch.no_grad()
def predict_real150(model, device, Ps=C.P_TEST, flips=True, cache_dir=None, log=print):
    rows = []
    t0 = time.time()
    for i in range(150):
        # the exact crops the annotators count on (common/rectify.py), else the old cache
        p = os.path.join(TOOL_CROPS, f"{i}.jpg")
        if not os.path.exists(p):
            p = os.path.join(cache_dir, f"{i}.png") if cache_dir else None
        crop = cv2.imread(p) if p and os.path.exists(p) else fd.load_rectified(i)
        crop = C.cap_side(crop, 1024)
        o = C.predict_crop(model, crop, device, Ps=Ps, flips=flips)
        rows.append(dict(image=i, pred_h=o["h"], pred_v=o["v"], h_iqr=o["h_iqr"], v_iqr=o["v_iqr"],
                         pred_weave=C.WEAVE_CLASSES[int(np.argmax(o["weave_probs"]))]))
        if (i + 1) % 50 == 0:
            log(f"    real150 {i + 1}/150  {time.time() - t0:.0f}s")
    return pd.DataFrame(rows)


def scatter_plot(legacy_df, synth_df, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(9, 4.2))
    if synth_df is not None:
        a = ax[0]
        a.scatter(synth_df.h_count, synth_df.pred_h, s=6, alpha=.5, label="h (vertical yarns)")
        a.scatter(synth_df.v_count, synth_df.pred_v, s=6, alpha=.5, label="v (horizontal yarns)")
        lim = [20, 220]
        a.plot(lim, lim, "k--", lw=.8)
        a.set_xlim(lim), a.set_ylim(lim)
        a.set_xlabel("true threads / inch"), a.set_ylabel("predicted")
        a.set_title("synthetic held-out"), a.legend(fontsize=8)
    if legacy_df is not None:
        a = ax[1]
        pa, pb, _ = C.best_assignment(legacy_df.pred_h, legacy_df.pred_v, legacy_df.warp, legacy_df.weft)
        a.scatter(legacy_df.warp, legacy_df.pred_h, marker="o", label="warp vs h (axis-matched)")
        a.scatter(legacy_df.weft, legacy_df.pred_v, marker="s", label="weft vs v (axis-matched)")
        a.scatter(legacy_df.warp, pa, marker="x", c="k", s=20, label="warp, best-assign.")
        a.scatter(legacy_df.weft, pb, marker="+", c="k", s=30, label="weft, best-assign.")
        for r in legacy_df.itertuples():
            a.annotate(r.name, (r.warp, r.pred_h), fontsize=6)
        lim = [30, 140]
        a.plot(lim, lim, "k--", lw=.8)
        a.set_xlim(lim), a.set_ylim(lim)
        a.set_xlabel("hand count (threads / inch)"), a.set_ylabel("predicted")
        a.set_title("legacy 2019 images (zero-shot)"), a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--synth-val", default=None, help="folder or comma-separated folders")
    ap.add_argument("--synth-n", type=int, default=None)
    ap.add_argument("--legacy", action="store_true")
    ap.add_argument("--legacy-modes", default="detect,raw",
                    help="comma list of legacy window modes: detect (shared frame detector), raw "
                         "(whole image = 1 inch), trim, vlm (canonical vlm/crops/legacy PNGs)")
    ap.add_argument("--real150", action="store_true")
    ap.add_argument("--real-cache", default=os.path.join(os.path.dirname(__file__), "data", "rectified"))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--fast", action="store_true", help="single scale, no flips (quicker)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = C.get_device(a.device)
    model = C.load_ckpt(a.ckpt, dev)
    Ps, fl = ((768,), False) if a.fast else (C.P_TEST, True)
    summary = {"ckpt": a.ckpt, "Ps": list(Ps), "flips": fl}
    sdf = ldf = None
    for sv in (a.synth_val.split(",") if a.synth_val else []):
        tag = os.path.basename(os.path.normpath(sv))
        d, m, br = eval_synth(model, sv, dev, a.synth_n, Ps=(768,), flips=False)
        d.to_csv(os.path.join(a.out, f"{tag}_predictions.csv"), index=False)
        rt, rp = np.log(d.h_count / d.v_count), np.log(d.pred_h / d.pred_v)
        m["logratio_corr"] = float(np.corrcoef(rt, rp)[0, 1])
        m["logratio_std_true"], m["logratio_std_pred"] = float(rt.std()), float(rp.std())
        m["logmean_corr"] = float(np.corrcoef(np.log(d.h_count * d.v_count), np.log(d.pred_h * d.pred_v))[0, 1])
        summary[tag] = m
        summary[tag + "_breakdown"] = br
        print(f"synthetic held-out ({tag}):", json.dumps(m, indent=1))
        if sdf is None:
            sdf = d
    if a.legacy:
        for k, mode in enumerate(a.legacy_modes.split(",")):
            d, res = eval_legacy(model, dev, Ps=Ps, flips=fl, mode=mode)
            d.to_csv(os.path.join(a.out, f"legacy_predictions_{mode}.csv"), index=False)
            summary[f"legacy_{mode}"] = res
            print(f"--- legacy ({mode}) ---")
            print(d.round(1).to_string())
            for sub in ("all13", "clear_no_range"):
                for key in ("axis_matched", "best_assignment"):
                    m = res[sub][key]
                    print(f"  {sub:15s} {key:16s} n={m['n']} warp MAE {m['MAE_h']:.1f} ({m['MAPE_h']:.1f}%, "
                          f"{m['n10_h']}/{m['n']} <=10%)  weft MAE {m['MAE_v']:.1f} ({m['MAPE_v']:.1f}%, "
                          f"{m['n10_v']}/{m['n']} <=10%)  MAPE {m['MAPE']:.1f}")
            if k == 0:
                ldf = d
    if a.real150:
        rdf = predict_real150(model, dev, Ps=Ps, flips=fl, cache_dir=a.real_cache)
        rdf.to_csv(os.path.join(a.out, "real150_predictions.csv"), index=False)
    if sdf is not None or ldf is not None:
        scatter_plot(ldf, sdf, os.path.join(a.out, "scatter.png"))
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
