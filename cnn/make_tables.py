#!/usr/bin/env python
"""Markdown tables for RESULTS.md from saved evaluation outputs (no recomputation
of model predictions; metrics recomputed from the saved per-image CSVs).

    python make_tables.py
"""
import json
import os

import numpy as np
import pandas as pd

import cnn_common as C

HERE = os.path.dirname(os.path.abspath(__file__))
HARNESS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "methods", "legacy_2019", "run1", "results.csv")

MODELS = [("CNN v1 recipe (8 min, isotropic, v1 data)", "runs/synth_resnet18/eval_final"),
          ("CNN probe (6 min/250 steps, aniso+ratio, v1 data)", "runs/probe_aniso_v1data/eval_final"),
          ("**CNN final (35 min, aniso+ratio, v1+v2 data)**", "runs/synth_r18_v2mix_aniso/eval_final")]


def fmt(m, ax):
    return f"{m['MAE_' + ax]:.1f} / {m['MAPE_' + ax]:.1f}% / {m['n10_' + ax]}"


def rows_for(name, ph, pv, gw, gf, mask=None):
    ph, pv, gw, gf = map(lambda a: np.asarray(a, float), (ph, pv, gw, gf))
    if mask is not None:
        ph, pv, gw, gf = ph[mask], pv[mask], gw[mask], gf[mask]
    am = C.count_metrics(ph, pv, gw, gf)
    pa, pb, sw = C.best_assignment(ph, pv, gw, gf)
    ba = C.count_metrics(pa, pb, gw, gf)
    return (f"| {name} | {am['n']} | {fmt(am, 'h')} | {fmt(am, 'v')} | {am['MAPE']:.1f}% | "
            f"{fmt(ba, 'h')} | {fmt(ba, 'v')} | {ba['MAPE']:.1f}% |")


def legacy_table(mode="detect", subset="all"):
    hdr = ("| method | n | warp MAE / MAPE / #≤10% (axis-matched) | weft (axis-matched) | mean MAPE | "
           "warp (best-assign.) | weft (best-assign.) | mean MAPE |\n|---|---|---|---|---|---|---|---|")
    out = [hdr]
    ref = pd.read_csv(os.path.join(HERE, MODELS[-1][1], f"legacy_predictions_{mode}.csv"))
    mask = None if subset == "all" else (~ref.unclear & ~ref.is_range).values
    gw, gf = ref.warp.values, ref.weft.values
    mu = float(np.mean(np.concatenate([gw, gf])))
    out.append(rows_for("constant = mean of the 26 GT counts (oracle-ish)", np.full(len(gw), mu),
                        np.full(len(gw), mu), gw, gf, mask))
    if os.path.exists(HARNESS):
        h = pd.read_csv(HARNESS)
        h = h[(h.variant == "precropped") & (h.seed == 0) & h.gt_usable.astype(bool)].copy()
        h["name"] = h.key
        h = ref[["name"]].merge(h, on="name", how="left")
        out.append(rows_for("2019 final counter (doc3 dot map + count.py), faithful reconstruction",
                            h.pred_warp_per_inch, h.pred_weft_per_inch, gw, gf, mask))
        rgb = h.xlsx_rgb_weft_pred.values.astype(float)
        e = np.abs(rgb - gf) if mask is None else np.abs(rgb - gf)[mask]
        g = gf if mask is None else gf[mask]
        out.append(f"| 2019 'RGB irregularity' (test.py, weft only) | {len(g)} | – | {e.mean():.1f} / "
                   f"{100 * (e / g).mean():.1f}% / {(e / g <= 0.1).sum()} | – | – | – | – |")
    lat = os.path.join(HERE, "countgd", "results", "lattice_fft_legacy13.csv")
    if os.path.exists(lat):
        L = ref[["name"]].merge(pd.read_csv(lat), on="name")
        out.append(rows_for("FFT lattice, strongest row/col peak (CountGD exemplar source)", L.fx, L.fy, gw, gf, mask))
    for name, d in MODELS:
        p = os.path.join(HERE, d, f"legacy_predictions_{mode}.csv")
        if os.path.exists(p):
            m = ref[["name"]].merge(pd.read_csv(p), on="name")
            out.append(rows_for(name, m.pred_h, m.pred_v, gw, gf, mask))
    gm = np.sqrt(gw * gf)
    out.append(rows_for("(bound) any isotropic predictor, even with the true geo-mean", gm, gm, gw, gf, mask))
    return "\n".join(out)


def synth_table():
    out = ["| model | val set (n) | MAE | MAPE | MdAPE | acc@10% | log(h/v) r | pred / true log-ratio std | weave acc |",
           "|---|---|---|---|---|---|---|---|---|"]
    for name, d in MODELS:
        p = os.path.join(HERE, d, "summary.json")
        if not os.path.exists(p):
            continue
        s = json.load(open(p))
        for tag in ("synth_val", "synth_val_v2"):
            if tag in s:
                m = s[tag]
                out.append(f"| {name} | {tag} ({m['n']}) | {m['MAE']:.1f} | {m['MAPE']:.1f}% | {m['MdAPE']:.1f}% | "
                           f"{m['acc@10%']:.1f}% | {m['logratio_corr']:.2f} | {m['logratio_std_pred']:.3f} / "
                           f"{m['logratio_std_true']:.3f} | {m['weave_acc']:.1f}% |")
    return "\n".join(out)


def window_table():
    out = ["| legacy window | axis-matched mean MAPE | best-assign. mean MAPE | geo-mean MAPE | log-ratio r (n=13) |",
           "|---|---|---|---|---|"]
    s = json.load(open(os.path.join(HERE, MODELS[-1][1], "summary.json")))
    for mode in ("detect", "raw", "trim"):
        r = s.get(f"legacy_{mode}")
        if r:
            out.append(f"| {mode} | {r['all13']['axis_matched']['MAPE']:.1f}% | {r['all13']['best_assignment']['MAPE']:.1f}% | "
                       f"{r['geomean_density']['MAPE']:.1f}% | {r['logratio_corr_axis_matched']:.2f} |")
    return "\n".join(out)


def per_image():
    d = pd.read_csv(os.path.join(HERE, MODELS[-1][1], "legacy_predictions_detect.csv"))
    v1 = pd.read_csv(os.path.join(HERE, MODELS[0][1], "legacy_predictions_detect.csv"))
    out = ["| image | weave | flags | GT warp × weft | final CNN h × v | v1 CNN h × v | final weave (p_plain/p_twill) |",
           "|---|---|---|---|---|---|---|"]
    for r, q in zip(d.itertuples(), v1.itertuples()):
        fl = ",".join([x for x, b in (("unclear", r.unclear), ("range", r.is_range)) if b]) or "–"
        out.append(f"| {r.name} | {r.weave} | {fl} | {r.warp:.0f} × {r.weft:.0f} | {r.pred_h:.1f} × {r.pred_v:.1f} | "
                   f"{q.pred_h:.1f} × {q.pred_v:.1f} | {r.pred_weave} ({r.p_plain:.2f}/{r.p_twill:.2f}) |")
    return "\n".join(out)


def degradation_table():
    runs = [("v1", "runs/synth_resnet18/degradation"), ("probe", "runs/probe_aniso_v1data/degradation"),
            ("final", "runs/synth_r18_v2mix_aniso/degradation")]
    dfs = {k: pd.read_csv(os.path.join(HERE, p, "degradation_summary.csv")) for k, p in runs
           if os.path.exists(os.path.join(HERE, p, "degradation_summary.csv"))}
    keys = list(dfs)
    out = ["| degradation | " + " | ".join(f"{k}: axis / best / geo-mean MAPE" for k in keys) + " |",
           "|---|" + "---|" * len(keys)]
    base = dfs[keys[0]]
    for i, r in base.iterrows():
        lab = "clean (600 px/in)" if r.kind == "clean" else f"{r.kind} {r.val:g}"
        cells = []
        for k in keys:
            q = dfs[k].iloc[i]
            cells.append(f"{q.MAPE_axis:.1f} / {q.MAPE_best:.1f} / {q.MAPE_geomean:.1f}")
        out.append(f"| {lab} | " + " | ".join(cells) + " |")
    return "\n".join(out)


if __name__ == "__main__":
    print("## synthetic\n" + synth_table())
    print("\n## legacy all 13 (detect)\n" + legacy_table("detect", "all"))
    print("\n## legacy clear, no range (detect)\n" + legacy_table("detect", "clear"))
    print("\n## window modes\n" + window_table())
    print("\n## per image\n" + per_image())
    try:
        print("\n## degradation\n" + degradation_table())
    except Exception as e:  # sweep may still be running
        print("degradation table unavailable:", e)
