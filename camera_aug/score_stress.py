"""score_stress.py -- PickGlass-Stress results on the legacy-13 (ours, log-Gabor, CNN, VLMs).

Scoring = paper/tools/legacy_unified.py's 'axis' rule: h (vertical yarns) <-> warp, v <-> weft,
13 images x 2 counts; APE = 100 |pred - label| / label; label = hand count x the render's exact
radial-distortion factor (stress/legacy13/labels.csv; 'phys' columns use the hand count unchanged).
Conditions with 3 renders (reps) are averaged over reps per count (the CI resamples IMAGES, keeping
both axes and all reps of an image together; 10 000 percentile-bootstrap resamples).
Paired vs clean (same method on the canonical crop): dMAPE with a paired image bootstrap CI, Wilcoxon
signed-rank on the 13 per-image mean APE differences, and the label-free drift
|pred / pred_clean - 1| (median over counts).  Unparsed VLM answers are not imputed (n shown).

    <venv2>/bin/python score_stress.py [--pick-harshest]
    -> results/stress_legacy13_metrics.csv, results/stress_legacy13_per_count.csv,
       ../paper/tables/camera_stress_legacy13.csv / .tex, results/stress_harshest.json (--pick-harshest)
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(HERE, "results")
TAB = os.path.join(ROOT, "paper", "tables")
B = 10_000
SEED = 20261007
CAMERA_CONDS = ["budget_binned_12mp", "budget_fullres_50mp", "midrange_binned", "flagship", "legacy_16mp_2019",
                "budget_lowlight", "budget_whatsapp", "legacy_quarter"]
ORDER = ["clean", "clean_repeat1", "clean_repeat2", "noop_store", "noop_q75"] + CAMERA_CONDS + ["budget+glare0.05", "budget+glare0.3",
                                                              "budget+gblur3", "budget+gblur6"]
NAMES = {"clean": "clean (canonical crop)", "clean_repeat1": "clean, VLM repeat 1",
         "clean_repeat2": "clean, VLM repeat 2", "noop_store": "no-op: crop-store re-encode (q92)",
         "noop_q75": "no-op: q75 resave + store", "budget_binned_12mp": "Budget 12.5 MP (binned)",
         "budget_fullres_50mp": "Budget 50 MP full-res", "midrange_binned": "Mid-range (binned, OIS)",
         "flagship": "Flagship (12-24 MP, OIS)", "legacy_16mp_2019": "Legacy 16 MP (2019, ours)",
         "budget_lowlight": "Budget + low light", "budget_whatsapp": "Budget + messaging (standard)",
         "legacy_quarter": "Legacy + q75 x0.25 resave", "budget+glare0.05": "Budget + glare 5 %",
         "budget+glare0.3": "Budget + glare 30 %", "budget+gblur3": "Budget + defocus 3 u",
         "budget+gblur6": "Budget + defocus 6 u"}
METHODS = {"ours": "Ours (PG-Lattice)", "gabor": "Log-Gabor bank", "cnn": "CNN, synthetic only",
           "gpt-6.1-sol": "GPT-6.1-sol (P1)", "gemini-3.8-flash": "Gemini 3.8 Flash (P1)"}


def load_preds():
    frames = []
    for f in ("stress_legacy13_classical.csv", "stress_legacy13_cnn.csv", "stress_legacy13_vlm.csv"):
        p = os.path.join(RES, f)
        if os.path.exists(p):
            d = pd.read_csv(p, dtype={"source_id": str})
            if "vlm_repeat" in d:   # stress renders: repeat 0; clean: repeat 0 + repeats 1-2 as 'clean_repeat<k>'
                d = d[(d.vlm_repeat == 0) | d.condition.str.startswith("clean_repeat")]
                if "bench_version" in d:   # only the current benchmark version (v1 = superseded capped renders)
                    d = d[d.bench_version.isin(["clean", "v2"])]
            frames.append(d[["source_id", "condition", "rep", "estimator", "h", "v"]])
    return pd.concat(frames, ignore_index=True)


def per_count(pred, lab):
    """One row per (method, condition, rep, image, axis)."""
    clean_lab = lab.drop_duplicates("source_id").set_index("source_id")[["h_src", "v_src"]]
    L = lab[["source_id", "condition", "rep", "h_label", "v_label", "h_src", "v_src"]]
    cl = pd.DataFrame({"source_id": clean_lab.index, "condition": "clean", "rep": 0,
                       "h_label": clean_lab.h_src.values, "v_label": clean_lab.v_src.values,
                       "h_src": clean_lab.h_src.values, "v_src": clean_lab.v_src.values})
    L = pd.concat([L, cl] + [cl.assign(condition=f"clean_repeat{k}") for k in (1, 2)], ignore_index=True)
    m = pred.merge(L, on=["source_id", "condition", "rep"], how="inner")
    rows = []
    for ax, lc, pc in (("warp", "h_label", "h"), ("weft", "v_label", "v")):
        t = m[["estimator", "condition", "rep", "source_id"]].copy()
        t["axis"] = ax
        t["pred"] = m[pc].astype(float)
        t["true"] = m[lc].astype(float)
        t["true_phys"] = m["h_src" if ax == "warp" else "v_src"].astype(float)
        rows.append(t)
    d = pd.concat(rows, ignore_index=True)
    d["ape"] = 100 * (d.pred - d.true).abs() / d.true
    d["ape_phys"] = 100 * (d.pred - d.true_phys).abs() / d.true_phys
    d["err_pct"] = 100 * (d.pred - d.true) / d.true
    clean = d[d.condition == "clean"].set_index(["estimator", "source_id", "axis"]).pred
    d["pred_clean"] = d.set_index(["estimator", "source_id", "axis"]).index.map(clean).values
    d["drift"] = 100 * (d.pred / d.pred_clean - 1)
    return d


def img_table(g):
    """per image: mean APE over its counts (both axes, all reps), n counts, n within 10 %."""
    ok = g[np.isfinite(g.ape)]
    t = ok.groupby("source_id").agg(ape=("ape", "mean"), ape_phys=("ape_phys", "mean"), n=("ape", "size"),
                                    w10=("ape", lambda s: (s <= 10).mean()))
    return t


def boot(vals_by_img, weights, idx):
    return (vals_by_img[idx] * weights[idx]).sum(1) / weights[idx].sum(1)


def score(d):
    import zlib
    rows = []
    for (est, cond), g in d.groupby(["estimator", "condition"]):
        rng = np.random.default_rng([SEED, zlib.crc32(f"{est}|{cond}".encode())])   # order-independent CIs
        t = img_table(g)
        if not len(t):
            continue
        n_img = len(t)
        idx = rng.integers(0, n_img, size=(B, n_img))
        w = t.n.to_numpy(float)
        a = t.ape.to_numpy()
        bs = boot(a, w, idx)
        ok = g[np.isfinite(g.ape)]
        reps = g.rep.nunique()
        r = dict(estimator=est, condition=cond, n_images=n_img, n_counts=int(len(ok) / max(1, reps)), reps=reps,
                 n_unparsed=int((~np.isfinite(g.ape)).sum()),
                 MAPE=float((a * w).sum() / w.sum()), MAPE_lo=float(np.percentile(bs, 2.5)),
                 MAPE_hi=float(np.percentile(bs, 97.5)), MAPE_phys=float((t.ape_phys * w).sum() / w.sum()),
                 MAE=float((ok.pred - ok.true).abs().mean()), bias_pct=float(ok.err_pct.mean()),
                 w10_pct=float(100 * (ok.ape <= 10).mean()), n_w10_of_26=float((ok.ape <= 10).sum() / reps),
                 drift_med_abs=float(ok.drift.abs().median()) if cond != "clean" else 0.0,
                 drift_p90_abs=float(ok.drift.abs().quantile(0.9)) if cond != "clean" else 0.0)
        # paired vs clean
        if cond != "clean":
            c = img_table(d[(d.estimator == est) & (d.condition == "clean")])
            j = t.join(c, rsuffix="_c", how="inner")
            if len(j) >= 3:
                diff = (j.ape - j.ape_c).to_numpy()
                ww = j.n.to_numpy(float)
                ii = rng.integers(0, len(j), size=(B, len(j)))
                bd = (diff[ii] * ww[ii]).sum(1) / ww[ii].sum(1)
                r.update(dMAPE=float((diff * ww).sum() / ww.sum()), dMAPE_lo=float(np.percentile(bd, 2.5)),
                         dMAPE_hi=float(np.percentile(bd, 97.5)),
                         p_wilcoxon_img=float(stats.wilcoxon(diff).pvalue) if np.any(diff != 0) else 1.0,
                         n_paired=len(j))
        rows.append(r)
    m = pd.DataFrame(rows)
    m["_o"] = m.condition.map({c: i for i, c in enumerate(ORDER)})
    m["_e"] = m.estimator.map({e: i for i, e in enumerate(METHODS)})
    return m.sort_values(["_e", "_o"]).drop(columns=["_o", "_e"])


def pick_harshest(m):
    sub = m[m.estimator.isin(["ours", "gabor", "cnn"]) & m.condition.isin(CAMERA_CONDS)]
    piv = sub.pivot(index="condition", columns="estimator", values="dMAPE")
    piv["mean_dMAPE"] = piv.mean(1)
    piv = piv.sort_values("mean_dMAPE", ascending=False)
    harsh = list(piv.index[:2])
    json.dump({"harshest": harsh, "rule": "largest mean paired dMAPE vs clean over ours, log-Gabor and CNN among the "
               "8 camera conditions; fixed before any VLM call", "table": piv.round(2).reset_index().to_dict("records")},
              open(os.path.join(RES, "stress_harshest.json"), "w"), indent=1)
    print("harshest (classical + CNN):", harsh)
    print(piv.round(2).to_string())
    return harsh


def fmt(x, nd=1):
    return "--" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


def latex(m, path):
    """Compact table: MAPE [95 % CI] per method x condition (camera conditions + no-op + clean)."""
    conds = ["clean", "noop_q75"] + CAMERA_CONDS + ["budget+glare0.3", "budget+gblur6"]
    ests = [e for e in METHODS if e in set(m.estimator)]
    L = [r"\begin{table*}[t]", r"\centering\small", r"\setlength{\tabcolsep}{4pt}",
         r"\caption{PickGlass-Stress on the 13 hand-counted legacy crops: MAPE (\%) with 95\,\% image-bootstrap CI "
         r"and, in parentheses, the paired change vs.\ the clean crop. Camera conditions are averaged over 3 renders "
         r"per image (VLMs: 1 render, harshest two conditions only); $^{*}$ = paired 95\,\% CI excludes 0. Labels = "
         r"hand counts $\times$ the exact distortion factor of each render; u = 1/1000 inch.}", r"\label{tab:camera_stress}",
         r"\begin{tabular}{l" + "c" * len(ests) + "}", r"\toprule",
         "Condition & " + " & ".join(METHODS[e] for e in ests) + r" \\", r"\midrule"]
    for c in conds:
        cells = []
        for e in ests:
            r = m[(m.estimator == e) & (m.condition == c)]
            if not len(r):
                cells.append("--")
                continue
            r = r.iloc[0]
            s = f"{fmt(r.MAPE)} [{fmt(r.MAPE_lo)}, {fmt(r.MAPE_hi)}]"
            if c != "clean" and "dMAPE" in r and np.isfinite(r.get("dMAPE", np.nan)):
                sig = r"$^{*}$" if (r.dMAPE_lo > 0 or r.dMAPE_hi < 0) else ""
                s += f" ({'+' if r.dMAPE >= 0 else ''}{fmt(r.dMAPE)}{sig})"
            cells.append(s)
        L.append(NAMES[c].replace("%", r"\%").replace(" x0.25", r" $\times$0.25") + " & " + " & ".join(cells) + r" \\")
        if c in ("noop_q75", "legacy_16mp_2019", "legacy_quarter"):
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    open(path, "w").write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pick-harshest", action="store_true")
    a = ap.parse_args()
    lab = pd.read_csv(os.path.join(HERE, "stress", "legacy13", "labels.csv"), dtype={"source_id": str})
    d = per_count(load_preds(), lab)
    d.to_csv(os.path.join(RES, "stress_legacy13_per_count.csv"), index=False)
    m = score(d)
    m.to_csv(os.path.join(RES, "stress_legacy13_metrics.csv"), index=False)
    cols = ["estimator", "condition", "n_counts", "reps", "n_unparsed", "MAPE", "MAPE_lo", "MAPE_hi", "dMAPE",
            "dMAPE_lo", "dMAPE_hi", "p_wilcoxon_img", "w10_pct", "drift_med_abs", "MAPE_phys"]
    cols = [c for c in cols if c in m]
    print(m[cols].round(2).to_string(index=False))
    if a.pick_harshest:
        pick_harshest(m)
    out = m.copy()
    out.insert(2, "condition_name", out.condition.map(NAMES))
    out.insert(1, "method", out.estimator.map(METHODS))
    out.to_csv(os.path.join(TAB, "camera_stress_legacy13.csv"), index=False)
    latex(m, os.path.join(TAB, "camera_stress_legacy13.tex"))
    print("wrote", os.path.join(TAB, "camera_stress_legacy13.csv"), "and .tex")


if __name__ == "__main__":
    main()
